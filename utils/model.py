import os
import ipdb, re, pytz, datetime, time, sys, pickle, glob, json, random, unidecode, unicodedata
from collections import Counter
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
from pprint import pprint
import torch
from torch import nn
from torch.nn import functional as F
from prettytable import PrettyTable
import tiktoken 

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.imports import print_runtime, d_head, vocab_size, visited_urls, batch_size, d_model, n_heads, n_layers, block_size, learning_rate, dropout, num_chars, encode, decode, world_size, add_gpu, tokenizer, max_acc_batch_size, num_chunked_batches, eval_iter

from utils.helpers import plotter, read_google_corpus_tokens, read_google_corpus_tensors

torch.manual_seed(1337)


class SingleHead(nn.Module):
    """ one-head self-attention"""
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # idx and targets are both (B,T) tensor of integers
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Computer attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) / np.sqrt(C)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)  # randomly prevent nodes from communicating

        # Perform weighted aggregation of the values 
        v = self.value(x) # (B, T, T)
        out = wei @ v # (B, T, C)
        
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel 
        num_heads: each self-attention head is a communication channel for the tokens in sequence
        d_head: dimension of k,q,v vectors
    """
    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead() for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            # as per the Attention paper, hidden layer size is 4x the input layer
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication (MultiHeadAttention) followed by computation (Feed Forward Network) """

    def __init__(self, d_model, n_heads):
        # d_model: embedding dimension, n_heads: the number of heads we'd like
        super().__init__() 
        self.sa = MultiHeadAttention(n_heads)
        self.ffwd = MLP(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # residual connections are a wonderful invention of modern science. Fork off and come back. 
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


class PositionalEncoding(nn.Module):  #@save
    """https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class DecoderModel(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()
        # each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model) # final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.device = device
        self.num_params = None
        self.footprint = None
        self.table = None
        self.specs = None
        self.print_hyperparams_to_stdout()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C) = (batch, time, vocab_size) = (4, 8, 65)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = token_embeddings + position_embeddings # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def print_hyperparams_to_stdout(self):
        num_params = 0
        for name, item in self.named_parameters():
            if len(item.shape) == 1:
                m = item.shape[0]
                num_params += m
            else:
                m, n = item.shape
                num_params += m * n

        if self.device == 0:
            if num_params < 1e3:
                str_num_params= (f"num_params: {num_params}")
            elif num_params < 1e6:
                str_num_params= (f"num_params: {int(num_params * 1e-3)} K")
            elif num_params < 1e9:
                str_num_params= (f"num_params: {int(num_params * 1e-6)} M")
            elif num_params < 1e12:
                str_num_params= (f"num_params: {(num_params * 1e-9):.3f} B")
            print(str_num_params)

            self.num_params = num_params
            self.footprint = num_params * next(self.parameters()).element_size()
            table = PrettyTable(['d_model', 'n_layers', 'n_heads', 'd_head', 'block_size', 'batch_size', 
                                 'acc_batch_size', 'learning_rate'])
            table.add_row([d_model, n_layers, n_heads, d_head, block_size, 
                           f'{batch_size*world_size}\n{batch_size}/GPU', max_acc_batch_size, learning_rate])

            self.table = table
            self.specs = str_num_params +\
                         f"\ntokenizer:{tokenizer}\nCUDA_VISIBLE_DEVICES:{os.environ['CUDA_VISIBLE_DEVICES']}"+\
                         f"\nworld_size:{world_size}"
            print(table)

        return num_params


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        if len(data) % block_size == 0:
            #  If no target token left for the last batch. 
            print(f'\n===> MyDataset: len(data) % block_size == 0')
            data = torch.cat(( data, torch.tensor(encode('.'), dtype=torch.long) ))

        total_batches = len(data) // block_size

        data2 = data[:total_batches * block_size]
        data2 = data2.view(total_batches, block_size)

        targets = data[1:total_batches*block_size+1]
        targets2 = targets.view(total_batches, block_size)

        self.data = data2
        self.targets = targets2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@torch.no_grad()
def perplexity(model, device):
    # idx is (B, T) array of indices in the current context

    s0 = time.time()
    data = torch.load('dataset/news_tensors/news-commentary-v6.en_000.pt')
    data = data.clone().detach().to(torch.long)

    mytestset = MyDataset(data, block_size=block_size)
    test_loader = torch.utils.data.DataLoader(dataset=mytestset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=DistributedSampler(mytestset))

    for i, (xb, yb) in enumerate(test_loader):
        logits, loss = model(xb, yb) # evaluate the loss
        if i == 0:
            ppl = torch.exp(loss)
        else:
            ppl = (ppl*i + torch.exp(loss)) / (i+1)
        if i == 10 or i == len(test_loader) - 2:
            break

    torch.distributed.all_reduce(ppl)
    ppl /= world_size
    ppl = ppl.detach().to('cpu')
    if device == 0:
        print(f'ppl:{ppl:.2f}   {print_runtime(s0, False)}')
    return ppl


def generate_text(model, device, step=None, seed_text=''):
    if device != 0:
        return
    
    model.eval()
    s0 = time.time()
    if not seed_text:
        seed_text = 'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:'    
    seed_tokens = torch.as_tensor(encode(seed_text), device=device, dtype=torch.long).view(1,-1)
    out_tokens = generate_tokens(model, device, idx=seed_tokens).tolist()
    output = f''.join(decode(out_tokens))
    print(f'\n===> generate_text() step:{step} {print_runtime(s0, False)}')
    print(output)
    print('---' *30 + '\n')
    model.train()


@torch.no_grad()
def generate_tokens(model, device, idx, temperature=.5, max_new_tokens=200):
    # idx is (B, T) array of indices in the current context
    print(f'temperature: {temperature}')
    
    for _ in range(max_new_tokens):
        # get prediction 
        # crop idx to the last block_size tokens
        logits, _ = model(idx[:, -block_size:]) # logits.shape: (1, T, n_vocab) 

        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)

        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)

        # sample from the distribution
        if temperature == 0:
            idx_next = torch.argmax(probs)
            idx_next = torch.tensor([[idx_next]], device=device)
        else:
            idx_next = torch.multinomial(probs / temperature, num_samples=1) # (B, 1)

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    return idx[0]


class WarmupCosineAnnealing(torch.optim.lr_scheduler.LambdaLR):
    # https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    """ Linearly increases learning rate from 0.1 to 1 over `x0` training steps.
        Decreases learning rate from 1. to 0.1 over the next `x1 - x0` steps following a b/(x+a) hyperbolic curve.
        Stays constant at 0.1 after x1 steps.
    """

    def __init__(self, optimizer, num_tokens, x0, x1, last_epoch=-1):
        self.num_tokens = num_tokens
        self.x0 = x0 
        self.x1 = x1 
        super(WarmupCosineAnnealing, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, x):
        # .step() invokes this function through LambdaLR
        if x < self.x0:
            # linear warm up stage
            y = .9 * x / max(1, self.x0) + 0.1 
        elif self.x0 <= x < self.x1:
            # cosine annealing stage
            T = (self.x1 - self.x0) * 2
            A = .5 * .9
            y = A * np.cos(2 * np.pi * (x - self.x0) / T) + 0.55    
        elif self.x1 <= x:
            # constant 10% learning_rate
            y = .1

        return y


class WarmupHyperbolicDecay(torch.optim.lr_scheduler.LambdaLR):
    # https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    """ Linearly increases learning rate from 0.1 to 1 over `x0` training steps.
        Decreases learning rate from 1. to 0.1 over the next `x1 - x0` steps following a b/(x+a) hyperbolic curve.
        Stays constant at 0.1 after x1 steps.
    """

    def __init__(self, optimizer, num_tokens, x0, x1, last_epoch=-1):
        self.x0 = x0 * num_tokens
        self.x1 = x1 * num_tokens
        super(WarmupHyperbolicDecay, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, x):
        # .step() invokes this function through LambdaLR
        if x < self.x0:
            # linear warm up stage
            y = .9 * x / max(1, self.x0) + 0.1 
        elif self.x0 <= x < self.x1:
            # hyperbolic decay stage
            b = 9 / (self.x1 - self.x0)
            a = (self.x0 * (1 - b)) / b 
            y = 1 / (b * (x + a) - self.x0 + 1) 
        elif self.x1 <= x:
            # constant 10% of max learning_rate
            y = .1

        return y


# def get_batch_size(batch_size, device, step, x, x0):
#         Call it like this. 
#         dynamic_batch_size = get_batch_size(batch_size, device, step, num_tokens, x1//20)
#     """
#     if x < x0:
#         c = 0.10
#         m = (1 - c) / x0 
#         y = m * x + c
#     else:
#         y = 1
#     if device == 0:
#         print(f'get_batch_size:{device} step =', step, '  x =', x, '  x0 =', x0,  '  y=', y)
#     return int(y * batch_size)


def load_train_objs(vocab_size, device, learning_rate):
    # instantiate model
    model = DecoderModel(vocab_size, device)

    # Create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, lr=learning_rate)  
    
    optimizer.zero_grad(set_to_none=False)
    return model, optimizer


def get_batch(val_data, block_size, batch_size, device):
    print('D.', len(val_data) , block_size)
    assert len(val_data) >= block_size
    ix = torch.randint(0, len(val_data) - block_size, (batch_size,))

    xb = torch.stack([val_data[i:i+block_size] for i in ix])
    yb = torch.stack([val_data[i+1:i+block_size+1] for i in ix])

    xb = xb.to(device)
    yb = yb.to(device)

    return xb, yb


@torch.no_grad()  # prevent allocation of gradient nodes by telling torch we're never gonna call .backward()
def estimate_loss(model, val_loader, device):
    s0 = time.time()
    
    model.eval()
    val_loss = []
    for batch_no, (xb, yb) in enumerate(val_loader):
        xb, yb = xb.to(device), yb.to(device)
        logits, iloss = model(xb, yb) # evaluate the loss, logits shape = (B*T/world_size, vocab_size)
        iloss = iloss.mean() # take average across all available GPUs
        val_loss.append(iloss.item())
        
    model.train()

    return sum(val_loss) / len(val_loss)


def load_shakespeare(num_chars=0, filename='dataset/tiny_shakespeare.txt'):
    """ 338,025 tokens
    """
    with open(filename, 'r', encoding='utf-8') as f:
        train_data = f.read()

    return train_data, num_chars + len(train_data) * 0.9


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_ddp_model(model, device, step, dname='models'):
    torch.distributed.barrier()
    PATH = f'{dname}/chkpt_{step:05d}.pt'
    model.eval()
    s0 = time.time() 

    if device == 0:
        torch.save(model.state_dict(), PATH)
        print(f'Saved model "{PATH}"  {print_runtime(s0, False)}')
    torch.distributed.barrier()
    return


def load_model(model, PATH):
    model.load_state_dict(torch.load(PATH))
    model.eval()


def load_ddp_model_to_single_device(model, PATH):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from collections import OrderedDict

    state_dict = torch.load(PATH)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        name = key[7:] # remove `module.`
        new_state_dict[name] = value

    # Here, Model should be a model.DecoderModel object as desired
    model.load_state_dict(new_state_dict)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])

    xb = xb.to(device)
    yb = yb.to(device)
    
    return xb, yb


def train(device, model, optimizer, val_data, world_size):

    start = time.time()
    list_steps, list_losses, list_steps_val, list_losses_val, list_lr, list_secs = [[] for _ in range(6)]
    list_ppl_val = []
    step = 0
    sample_no = 0
    num_tokens = 0 
    idx_json = 0
    epoch = -1
    x0 = .375e6 * 100  # num_tokens at end of Linear warm up
    x1 = 260e6   # num_tokens at end of Cosine Annealing or Hyperbolic Decay
    list_steps, list_steps_val, list_losses, list_losses_val, list_lr, list_secs = [], [], [], [], [], []
    lr_scheduler = WarmupCosineAnnealing(optimizer, num_tokens, x0=x0, x1=x1)

    # tiny_shakespeare dataset:
    # val_data, _ = load_shakespeare()
    # train_data = val_data[:int(len(val_data) * 0.9)]
    #val_data = val_data[int(len(val_data) * 0.9):]
    #val_data = torch.tensor(encode(val_data))
    # train_data = torch.tensor(encode(train_data))

    val_data, idx_json = read_google_corpus_tokens(device, idx_json)
    if device == 0: 
        print(f'len(val_data):{len(val_data)}')
        print(f'batch_size:{batch_size}')
        print(f'block_size:{block_size}')
        print(f'max_acc_batch_size:{max_acc_batch_size}')
        print(f'num_chunked_batches:{num_chunked_batches}\n')

    # load validation data into DataLoader object
    #myvalset = MyDataset(val_data, block_size)
    #val_loader = torch.utils.data.DataLoader(dataset=myvalset,
    #                                         batch_size=batch_size,
    #                                         shuffle=False,
    #                                         sampler=DistributedSampler(myvalset))

    # compute losses on the randomly initialized model
    list_steps_val.append(step)
    list_losses_val.append(None)
    list_ppl_val.append(perplexity(model, device))

    plotter(model, device, list_steps, list_losses, list_lr, list_ppl_val, list_steps_val, list_losses_val, list_secs, start)
    s0 = time.time()

    # train-loop
    while epoch < 5e3:
        # at each epoch another 0.9 billion tokens are loaded.
        epoch += 1
        if device == 0: 
            print(f'\n\nepoch:{epoch}   num_chunked_batches:{num_chunked_batches}')
        train_data, idx_json = read_google_corpus_tensors(device, idx_json)

        # load train data into DataLoader object
        mytrainset = MyDataset(train_data, block_size=block_size)
        train_loader = torch.utils.data.DataLoader(dataset=mytrainset,
                                                   batch_size=(num_chunked_batches*batch_size),
                                                   shuffle=False,
                                                   sampler=DistributedSampler(mytrainset))

        s1 = time.time()
        for batch_no, (xb_step, yb_step) in enumerate(train_loader):
            model.train()
            xb_step = xb_step.to(device)  #  each device ==> (34 * 4, 512)
            yb_step = yb_step.to(device)  
            total_size = 0
            for i in range(num_chunked_batches):
                xb = xb_step[i*batch_size : (i+1)*batch_size]  # ecah device ==> (4, 512) 
                yb = yb_step[i*batch_size : (i+1)*batch_size]
                num_tokens += len(xb) * block_size * world_size
                total_size += len(xb) * block_size * world_size
                if device == 0:
                    print('.', end='')
                
                logits, loss = model(xb, yb) # evaluate the loss
                loss.backward() # adds to the gradients

            if total_size != max_acc_batch_size:
                print(f'device:{device}:  total_size != max_acc_batch_size.  total_size={total_size}')

            step += 1
            optimizer.step() # Updates the weights:  w = w - grad * lr
            optimizer.zero_grad(set_to_none=False)
            lr_scheduler.step(num_tokens)
            list_lr.append(optimizer.param_groups[-1]['lr'])
            torch.distributed.all_reduce(loss)
            loss /= world_size
            list_steps.append(step)
            list_losses.append(loss.item())
            if device == 0: 
                print(f'step:{step}, loss:{list_losses[-1]:.2f}')

            # total number of tokens ingested by all GPUs
            list_secs.append(time.time() - s1)
            s1 = time.time()

            if step % eval_iter == 0: # 5 steps
                # evaluate at fixed intervals
                if device == 0:
                    print(f'\nstep:{step}   loss:{loss.item():.2f}   num_tokens:{num_tokens/1e6:.2f} million  '+
                          f'batch_no:{batch_no:2d}/{len(train_loader)},  '+
                          f'Wall Time (one step):{list_secs[-1]:.1f} sec')

                #list_losses_val.append(estimate_loss(model, val_loader, device))
                list_steps_val.append(step)
                list_losses_val.append(None)
                list_ppl_val.append(perplexity(model, device))

            if step % (eval_iter * 10) == 0:  # 50 steps
                plotter(model, device, list_steps, list_losses, list_lr, list_ppl_val, list_steps_val, 
                        list_losses_val, list_secs, start)

            if step % (eval_iter * 100) == 0: # 500 steps
                generate_text(model, device, step)
                save_ddp_model(model, device, step)

        if device == 0:
            print(f'\n..... step:{step}: {len(train_data)*1e-6:.2f} M tokens. '+
                  f'num_pages {len(visited_urls):02d}: Total time:{print_runtime(start, False)[1:-1]}', end='\n\n')



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        