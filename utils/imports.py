# Hyperparameters for transformer model
d_model = 1728
n_heads = 18
n_layer = 24
learning_rate = 2e-4
batch_size = 70 # (B) # each GPU gets `batch_size/world_size` samples
block_size = 128 # (T) # maximum context length for predictions. Looks at 256 to predict 257
dropout = 0.0 # use 0.0 for pre-training. For fine-tuning maybe 0.1 or 0.2
max_iters = 10000
eval_steps = 30

# --------------------------------------

num_chars = 0 
visited_urls = dict()
add = 5e6 # number of characters to be crawled 
d_head = int(d_model / n_heads)

assert d_model / n_heads % 1 == 0

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
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import tiktoken 
from tiktoken_ext.openai_public import ENCODING_CONSTRUCTORS

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# ------------------------------------------------
str_vocab = '\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
vocab = set(str_vocab)

encFunc = ENCODING_CONSTRUCTORS['gpt2']
encDict = encFunc()
enc = tiktoken.Encoding(encDict['name'],
                        pat_str=encDict['pat_str'],
                        mergeable_ranks=encDict['mergeable_ranks'],
                        special_tokens=encDict['special_tokens' ])

vocab_size = enc.n_vocab
# vocab_size = len(vocab)
encode = enc.encode
decode = enc.decode
world_size = torch.cuda.device_count()
if batch_size % world_size > 0:
    print(f'===>batch_size % world_size = {batch_size % world_size}. '+
          f'batch_size will be clipped to {world_size * (batch_size // world_size)}')
batch_size //= world_size

# A simple character based tokenizer.
# stoi = {s:i for i,s in enumerate(str_vocab)}
# itos = {i:s for i,s in enumerate(str_vocab)}
# def encode(str_input):
#     return [stoi[c] for c in str_input]

# def decode(list_idx):
#     return ''.join([itos[i] for i in list_idx])
    


def print_runtime(start, printer=True):
    end = time.time()
    if printer:
        print(f'Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec')
        return None
    else:
        if int((end-start)//60) == 0:
            return f'({int((end-start)%60)} sec)'
        else:
            return f'({int((end-start)//60)} min {int((end-start)%60):2d} sec)'


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


new_links = ["https://www.wikipedia.org/wiki/David_Bowie"]

pylab.rcParams.update({'legend.fontsize': 'small',
                       'font.size'      : 14,
                       'figure.figsize' : (9, 3.5),
                       'axes.labelsize' : 'medium',
                       'axes.titlesize' : 'medium',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'medium',
                       'ytick.labelsize': 'medium'})

