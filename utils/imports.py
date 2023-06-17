# Hyperparameters for transformer model
batch_size = 64 # (B)
d_model = 384
n_heads = 6
n_layer = 6
block_size = 128 # (T) # maximum context length for predictions. Looks at 256 to predict 257
learning_rate = 3e-4
dropout = 0.0 # use 0.0 for pre-training. For fine-tuning maybe 0.1 or 0.2
max_iters = 1000000

# --------------------------------------

eval_num_samples = int((64 * 500 * 64) / batch_size) # sample_no

# --------------------------------------

num_chars = 0
visited_urls = dict()
add = 2.5e6 # number of tokens to be crawled 
# d_model = d_head * n_heads # (C) --each head is 64 dimensional
d_head = int(d_model / n_heads)


assert d_model / n_heads % 1 == 0
# set the visible GPUs for CUDA to use.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import ipdb, re, time, sys, pickle, glob, json, random, unidecode, unicodedata
from IPython.display import clear_output
from collections import Counter
from urllib.request import urlopen
from bs4 import BeautifulSoup

import numpy as np
from pprint import pprint
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch
from torch import nn
from torch.nn import functional as F
import tiktoken
from prettytable import PrettyTable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pylab.rcParams.update({'legend.fontsize': 'small',
                       'font.size'      : 14,
                       'figure.figsize' : (9, 3.5),
                       'axes.labelsize' : 'medium',
                       'axes.titlesize' : 'medium',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'medium',
                       'ytick.labelsize': 'medium'})

print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")


def print_runtime(start, printer=True):
    end = time.time()
    if printer:
        print(f'Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec')
        return None
    else:
        return f' (...Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec)'

    
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
    
