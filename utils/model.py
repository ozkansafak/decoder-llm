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

from utils.imports import print_runtime, count_parameters, d_head, vocab_size, vocab, new_links, visited_urls, batch_size, d_model, n_heads, n_layer, block_size, learning_rate, dropout, max_iters, eval_steps, num_chars, add, encode, decode, world_size

from utils.helpers import load_val_data, extract_single_url, get_links, shave, decompose_divs, plotter, clean_up, ptxt, crawl_wiki_data

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


class DecoderModel(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()
        # each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model) # final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.device = device
        self.num_params = None
        self.footprint = None
        self.print_hyperparams_to_stdout()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C) = (batch, time, vocab_size) = (4, 8, 65)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        #position_embeddings = 0
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
            loss = loss.mean()  

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
                print(f"num_params: {num_params}")
            elif num_params < 1e6:
                print(f"num_params: {int(num_params * 1e-3)} K")
            elif num_params < 1e9:
                print(f"num_params: {int(num_params * 1e-6)} M")
            elif num_params < 1e12:
                print(f"num_params: {(num_params * 1e-9):.3f} B")

            self.num_params = num_params
            self.footprint = num_params * next(self.parameters()).element_size()
            print(f'model.footprint: {self.footprint * 1e-6:.2f} MB')
            table = PrettyTable(['d_model', 'n_layer', 'n_heads', 'd_head', 'block_size', 'batch_size', 'learning_rate'])
            table.add_row([d_model, n_layer, n_heads, d_head, block_size, 
                           f'{batch_size* world_size}\n({batch_size} per GPU)', learning_rate])
            print(table)

        return num_params


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, block_size):
        if len(train_data) % block_size == 0:
            # if train_set doesn't have an extra token as target for the last sample, put a dot (.) artificially.
            print(f'\n ===> MyDataset.__init__(): len(train_data) % block_size == 0\n')
            train_data = torch.cat((train_data, torch.tensor([encode('.')], dtype=torch.long)))

        total_batches = len(train_data)//block_size
        train_data2 = train_data[:total_batches * block_size]
        train_data2 = train_data2.view(total_batches, block_size)
        train_targets = train_data[1:total_batches*block_size+1]
        train_targets2 = train_targets.view(total_batches, block_size)

        self.data = train_data2
        self.targets = train_targets2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def generate_text(model, device, step, loss):
    if device !=0:
        return
    s0 = time.time()

    seed_text = 'Into the flood again'    
    seed_tokens = torch.as_tensor(encode(seed_text), device=device, dtype=torch.long).view(1,-1)
    out_tokens = generate_tokens(model, device, idx=seed_tokens).tolist()
    output = f''.join(decode(out_tokens))
    print(f'===>  Text Generation step:{step}, loss:{loss:.2f} {print_runtime(s0, False)}')
    print(output)
    print('---' *30 + '\n')


@torch.no_grad()
def generate_tokens(model, device, idx, temperature=0.5, max_new_tokens=200):
    # idx is (B, T) array of indices in the current context

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


def load_train_objs(vocab_size, device, learning_rate):

    # instantiate model
    model = DecoderModel(vocab_size, device)
 
    # Create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # usually 3e-4 for bigger networks.

    return model, optimizer


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
    with open(filename, 'r', encoding='utf-8') as f:
        train_data = f.read()

    return train_data, num_chars + len(train_data) * 0.9


def train(device, model, optimizer, num_chars, val_data, world_size,
          list_num_tokens, list_losses, list_num_tokens_val, list_losses_val, eval_steps):

    start = time.time()
    step = 0
    sample_no = 0
    num_tokens = 0
    num_batches = 0

#     val_data, _ = load_shakespeare()
#     train_data = val_data[:int(len(val_data) * 0.9)]
#     val_data = val_data[int(len(val_data) * 0.9):]
#     val_data = torch.tensor(encode(val_data))

    myvalset = MyDataset(val_data, block_size)
    val_loader = torch.utils.data.DataLoader(dataset=myvalset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             sampler=DistributedSampler(myvalset))

    list_losses_val.append(estimate_loss(model, val_loader, device))
    list_num_tokens_val.append(num_tokens)
    plotter(device, list_num_tokens, list_losses, list_num_tokens_val, list_losses_val, savefig=True)

    s0 = time.time()
    while step < max_iters:
        # crawl a new batch of wiki pages
        train_data, num_chars = crawl_wiki_data(device, new_links, visited_urls, num_chars, add)
        mytrainset = MyDataset(train_data, block_size)
        train_loader = torch.utils.data.DataLoader(dataset=mytrainset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   sampler=DistributedSampler(mytrainset))

        for batch_no, (xb, yb) in enumerate(train_loader):
            step += 1
            # xb, yb is per GPU. Each GPU gets a different (xb,yb) but they're of same size.
            xb, yb = xb.to(device), yb.to(device)
            mb1 = xb.element_size() * batch_size * block_size * (block_size + 1) / 2 * 1e-6
            mb2 = yb.element_size() * batch_size * block_size * (block_size + 1) / 2 * 1e-6
            logits, loss = model(xb, yb) # evaluate the loss
            if device == 0 and ((batch_no + 1) % 10 == 0 or (batch_no + 1) == 1):
                print(f'batch_no:{batch_no+1:3d} of {len(train_loader)}, '+
                      f'loss:{loss.item():.2f}, Memory per GPU={mb1+mb2:.2f} MB + '+
                      f'{model.module.footprint*1e-6:.2f} MB')

            loss = loss.mean() # take average across the 8 GPUs
            optimizer.zero_grad(set_to_none=False)
            loss.backward() # get the gradients with backprop.
            optimizer.step() # apply the gradient on the network parameters.
            torch.distributed.all_reduce(loss)
            loss /= world_size
            list_losses.append(loss.item())

            # total number of tokens ingested by all GPUs
            num_tokens += block_size * len(xb) * world_size
            list_num_tokens.append(num_tokens)

            # evaluate at fixed intervals
            if device == 0 and step % eval_steps == 0:
                print('\n' + '---' *30)
                print(f'estimating loss... step:{step:3d} '+
                      f'train_data.device:{train_data.device}, val_data.device:{val_data.device} {print_runtime(s0, False)}')
                s0 = time.time()
                list_losses_val.append(estimate_loss(model, val_loader, device))
                list_num_tokens_val.append(num_tokens)
                plotter(device, list_num_tokens, list_losses, list_num_tokens_val, list_losses_val, savefig=True)
                if step % (eval_steps*4) == 0:
                    generate_text(model, device, step, list_losses[-1])

        del train_loader, mytrainset, train_data

        if device ==0:
            print(f'\ntrain() epoch finished. step:{step}: num_pages {len(visited_urls):02d}: '+
                  f'{"FINISHED " if step == max_iters else ""} {print_runtime(start, False)}', end='\n')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        