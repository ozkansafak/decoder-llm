# Hyperparameters for transformer model
batch_size = 128 # (B)
d_model = 768
n_heads = 24
n_layer = 12
learning_rate = 3e-5
block_size = 128 # (T) # maximum context length for predictions. Looks at 256 to predict 257
dropout = 0.0 # use 0.0 for pre-training. For fine-tuning maybe 0.1 or 0.2
max_iters = 100000

# --------------------------------------

eval_steps = 10

# --------------------------------------

num_chars = 0 
visited_urls = dict()
add = 2.5e6 # number of tokens to be crawled 
d_head = int(d_model / n_heads)

assert d_model / n_heads % 1 == 0


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# device = 'cuda:7'

# ------------------------------------------------
vocab = set('\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
list_vocab = sorted(vocab)
vocab_size = len(vocab)
_stoi = {c:i for i, c in enumerate(list_vocab)}
_itos = {i:c for i, c in enumerate(list_vocab)}
encode = lambda s: [_stoi[c] for c in s]  # takes in a string, output list of integers
decode = lambda inp: [_itos[i] for i in inp]  # input a list of integers, outputs a string


def print_runtime(start, printer=True):
    end = time.time()
    if printer:
        print(f'Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec')
        return None
    else:
        return f'({int((end-start)//60)} min {int((end-start)%60):2d} sec)'


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

new_links = ["https://www.wikipedia.org/wiki/David_Bowie"]
print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

pylab.rcParams.update({'legend.fontsize': 'small',
                       'font.size'      : 14,
                       'figure.figsize' : (9, 3.5),
                       'axes.labelsize' : 'medium',
                       'axes.titlesize' : 'medium',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'medium',
                       'ytick.labelsize': 'medium'})

