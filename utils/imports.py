# Hyperparamerters
n_layers = 12
d_model = 512
n_heads = 8
block_size = 512 # (T) # maximum context length for predictions.
batch_size_gpu = 24  # (B) # total batch_size summed across all GPUs
learning_rate = 6e-5
x0 = 0.375e9 # num_tokens at end of Linear warm up
x1 = 3e9  # num_tokens at end of Cosine Annealing or Hyperbolic Decay

max_acc_batch_size = int(0.5e6)
dropout = 0.0 # use 0.0 for pre-training. For fine-tuning maybe 0.1 or 0.2
tokenizer = 'gpt2'
eval_iter = 5
# --------------------------------------

num_chars = 0 
visited_urls = dict()
add_gpu = int(5e6) # number of tokens to be crawled 
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

# ------------------------------------------------------------------------------------------------
str_vocab = '\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
vocab = set(str_vocab)

assert tokenizer in ['gpt2', 'character']
    
world_size = torch.cuda.device_count()
batch_size = batch_size_gpu * torch.cuda.device_count()
if batch_size % world_size > 0:
    print(f'===> batch_size not a multiple of world_size: (batch_size % world_size = {batch_size % world_size}) '+
          f'batch_size will be clipped to {(batch_size // world_size) * world_size}')
    batch_size = (batch_size // world_size) * world_size

# this is batch_size (B) per gpu.
batch_size //= world_size
max_acc_batch_size = (max_acc_batch_size // (batch_size * block_size * world_size)) * (batch_size * block_size * world_size)  
num_chunked_batches = int(max_acc_batch_size / (batch_size * block_size * world_size))

_directory = "/data/home/osafak/code/mygpt/dataset/news_tensors"
_PATH = f"{_directory}/*.pt"
ls_pt = glob.glob(_PATH)
ls_pt.sort(key=lambda x: x.split("/")[-1])
def print_trainset_deets():
    """
        europarl-v6.en_000.pt        -- 59.34 million tokens
        news-commentary-v6.en_000.pt -- 4.81 million tokens  (used as validation set)
        news.2007.en.shuffled_000.pt -- 392.89 million tokens
        news.2008.en.shuffled_000.pt -- 900.00 million tokens
        news.2008.en.shuffled_001.pt -- 68.16 million tokens
        news.2009.en.shuffled_000.pt -- 900.00 million tokens
        news.2009.en.shuffled_001.pt -- 275.56 million tokens
        news.2010.en.shuffled_000.pt -- 458.62 million tokens
        news.2011.en.shuffled_000.pt -- 63.05 million tokens
    """

    if torch.distributed.get_rank() == 0:
        print('trainset:')
        for fname in ls_pt:
            data = torch.load(fname)
            print(f"         {fname.split('/')[-1]:28s} -- {len(data)/1e6:.2f} million tokens")


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


with open('dataset/scraped_urls.json', 'r') as f:
    # len(scraped_urls) = 2450704
    scraped_urls = json.load(f)
    random.shuffle(scraped_urls)
    
if tokenizer == 'gpt2':
    encFunc = ENCODING_CONSTRUCTORS['gpt2']
    encDict = encFunc()
    enc = tiktoken.Encoding(encDict['name'], pat_str=encDict['pat_str'], mergeable_ranks=encDict['mergeable_ranks'], 
                            special_tokens=encDict['special_tokens' ])
    vocab_size = enc.n_vocab
    encode = enc.encode
    decode = enc.decode
elif tokenizer == 'character':
    vocab_size = len(vocab)
    stoi = {s:i for i,s in enumerate(str_vocab)}
    itos = {i:s for i,s in enumerate(str_vocab)}
    def encode(str_input):
        return [stoi[c] for c in str_input]

    def decode(list_idx):
        return ''.join([itos[i] for i in list_idx])

pylab.rcParams.update({'legend.fontsize': 'small',
                   'font.size'      : 12,
                   'figure.figsize' : (9, 3.5),
                   'axes.labelsize' : 'small',
                   'axes.titlesize' : 'small',
                   'axes.grid'      : 'on',
                   'xtick.labelsize': 'small',
                   'ytick.labelsize': 'small'})

