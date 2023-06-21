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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


from utils.imports import print_runtime, count_parameters, d_head, vocab_size, vocab, list_vocab, new_links, visited_urls, batch_size, d_model, n_heads, n_layer, block_size, learning_rate, dropout, max_iters, eval_steps, num_chars, add, encode, decode

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

        print()
        if num_params < 1e3:
            print(f"num_params: {num_params}")
        elif num_params < 1e6:
            print(f"num_params: {int(num_params * 1e-3)}K")
        elif num_params < 1e9:
            print(f"num_params: {int(num_params * 1e-6)}M")
        elif num_params < 1e12:
            print(f"num_params: {int(num_params * 1e-9)}B")

        table = PrettyTable(['d_model', 'n_layer', 'n_heads', 'd_head', 'block_size', 'batch_size', 'learning_rate'])
        table.add_row([d_model, n_layer, n_heads, d_head, block_size, batch_size, learning_rate])
        print(table)

        return num_params


def generate_text(model, device, start=None):
    if start is None:
        start = time.time()
    print(f'===>  Text Generation: ')
    print(f''.join(decode(generate(model, 
                                   idx=torch.ones((1,1), device=device, dtype=torch.long) * 35, 
                                   max_new_tokens=400)[0].tolist()))) 
    print_runtime(start)
    print('---' *30)


def generate(model, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context

    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]

        # get prediction 
        logits, _ = model(idx_cond)

        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)

        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)

        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    return idx


@torch.no_grad()  # tells torch we're never gonna call .backward()
def estimate_loss(model, train_data, val_data,device):
    s0 = time.time()
    losses = {}
    print('\nestimating loss... ', end='')
    model.eval()
    for i, data in enumerate([train_data[-len(val_data):], val_data]):
        pivot = 0
        i_losses = [] 
        while pivot == 0 or len(xb) == batch_size:
            xb, yb, pivot = get_batch(data, device, pivot)
            logits, loss = model(xb, yb)
            i_losses.append(np.mean(loss.tolist()))
        losses['train' if i == 0  else 'val'] = np.mean(i_losses)

    mb1 = train_data.element_size() * train_data.nelement() * 1e-6
    mb2 = val_data.element_size() * val_data.nelement() * 1e-6
    print(f'train_loss:{losses["train"]:.4f}, val_loss:{losses["val"]:.4f},  Memory:{mb1+mb2}MB {print_runtime(s0, False)}')

    model.train()
    return losses


def load_train_objs(vocab_size, device, learning_rate):

    # instantiate model
    model = DecoderModel(vocab_size, device)
    model = model.to(device) # move model parameters to gpu if available
 
    # Create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # usually 3e-4 for bigger networks.

    return model, optimizer


def get_batch(data, device, pivot):
    """   todo: Take one page as input.
    """
    
    if len(data) - pivot < (batch_size * block_size):
        procured_batch_size = (len(data) - pivot) // block_size
        print(f' ==> get_batch: procured_batch_size = {procured_batch_size} out of {batch_size} requested')
        if procured_batch_size == 0:
            xb = torch.zeros((0, block_size), dtype=torch.long)
            yb = torch.zeros((0, block_size), dtype=torch.long)
            xb, yb = xb.to(device), yb.to(device)  # move data to gpu if available 
            pivot = 0
            return xb, yb, pivot
    
    ix = torch.arange(start=pivot, end=len(data) - block_size, step=block_size)[:batch_size]
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)  # move data to gpu if available
    
    pivot += batch_size * block_size
    return xb, yb, pivot


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, block_size):
        if len(train_data) % block_size == 0:
            # if train_set doesn't have an extra token as target for the last sample, synthetically put in 
            print(f'\n ===> MyDataset.__init__(): len(train_data) % block_size == 0\n')
            train_data = torch.cat((train_data, torch.tensor([3], dtype=torch.long)))
            
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


def train(model, optimizer, device, num_chars, val_data, 
          list_num_tokens, list_losses, list_num_tokens_eval, list_losses_eval, eval_steps, pend='\r'):

    start = time.time()
    step = 0
    sample_no = 0
    num_tokens = 0
    num_batches = 0

    while step < max_iters:
        step += 1

        # crawl a new batch of wiki pages
        train_data, num_chars = crawl_wiki_data(new_links, visited_urls, num_chars, add//10, printer=False)

        # wrap the data in DataLoader class. 
        mydataset = MyDataset(train_data, block_size)
        train_loader = torch.utils.data.DataLoader(
            dataset=mydataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        for batch_no, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            mb1 = xb.element_size() * xb.nelement() * 1e-6
            mb2 = yb.element_size() * yb.nelement() * 1e-6
            logits, loss = model(xb, yb) # evaluate the loss
            if (batch_no + 1) % 10 == 0 or (batch_no + 1) == 1:
                print(f'batch_no:{batch_no+1} of {len(train_loader)},  loss:{loss.item():.2f}, Memory:{mb1+mb2}MB ', end=pend)
            loss = loss.mean() # take average across the 8 GPUs
            optimizer.zero_grad(set_to_none=True)
            loss.backward() # get the gradients with backprop.
            optimizer.step() # apply the gradient on the network parameters.
            list_losses.append(loss.item())
            num_tokens += block_size * batch_size 
            list_num_tokens.append(num_tokens)

            # evaluate at fixed intervals
            if step % eval_steps == 0 and batch_no == (len(train_loader) - 1):
                print(f'step:{step:3d}')
                out = estimate_loss(model, val_data, val_data, device)
                list_losses_eval['train'].append(out['train'])
                list_losses_eval['val'].append(out['val'])
                list_num_tokens_eval.append(num_tokens)
                plotter(list_num_tokens, list_losses, list_num_tokens_eval, list_losses_eval, savefig=True)

            if step % (eval_steps * 4) == 0:
                generate_text(model, device) 

        print(f'step:{step:3d} num_pages:{len(visited_urls):02d}  '+
              f'{"FINISHED " if step == max_iters else ""} train():{print_runtime(start, False)[1:-1]}', end='\n')


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        










