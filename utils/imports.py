# Hyperparameters for transformer model
batch_size = 128 # (B)
block_size = 256 # (T) # maximum context length for predictions. Looks at 256 to predict 257
max_iters = 50000
d_head = 64
n_head = 6
n_layer = 6
dropout = 0.2 # 20% of nodes is disabled 
learning_rate = 3e-4

# --------------------------------------
eval_iters = 200
eval_interval = int((64 * 500) / batch_size)
eval_generate = 500
# --------------------------------------

num_chars = 0
visited_urls = dict()
add = 25e5 # number of tokens to be crawled 
d_model = d_head * n_head # (C) --each head is 64 dimensional

# set the visible GPUs for CUDA to use.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

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

assert int(d_model // n_head) - (d_model // n_head) == 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


pylab.rcParams.update({'legend.fontsize': 'small',
                       'font.size'      : 14,
                       'figure.figsize' : (9, 4),
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

    
