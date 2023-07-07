# Hyperparameters for transformer model
d_model = 768
n_heads = 12
n_layer = 12
learning_rate = 3e-5
batch_size = 64 # (B)
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
vocab = set('\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')

encFunc = ENCODING_CONSTRUCTORS['gpt2']
encDict = encFunc()
enc = tiktoken.Encoding(encDict['name'],
                        pat_str=encDict['pat_str'],
                        mergeable_ranks=encDict['mergeable_ranks'],
                        special_tokens=encDict['special_tokens' ])

vocab_size = enc.n_vocab
encode = enc.encode
decode = enc.decode



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
print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

pylab.rcParams.update({'legend.fontsize': 'small',
                       'font.size'      : 14,
                       'figure.figsize' : (9, 3.5),
                       'axes.labelsize' : 'medium',
                       'axes.titlesize' : 'medium',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'medium',
                       'ytick.labelsize': 'medium'})

