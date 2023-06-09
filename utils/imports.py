from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import ipdb, re, time, sys, os, pickle, glob, json, random, unidecode, unicodedata
from IPython.display import clear_output
from collections import Counter
from urllib.request import urlopen
from bs4 import BeautifulSoup

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import numpy as np
from pprint import pprint
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch
from torch import nn
from torch.nn import functional as F
import tiktoken


# Hyperparameters for torch model
batch_size = 64 # (B)
block_size = 256 # (T) # maximum context length for predictions. Looks at 256 to predict 257
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # (C) --every head is 64 dimensional
n_head = 6
n_layer = 6
dropout = 0.2 # 20% of nodes is disabled 
# ----------------------------


pylab.rcParams.update({'legend.fontsize': 'medium',
                       'font.size'      : 14,
                       'figure.figsize' : (18, 4),
                       'axes.labelsize' : 'medium',
                       'axes.titlesize' : 'medium',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'medium',
                       'ytick.labelsize': 'medium'})

print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")


def print_runtime(start, printer=True, end='\n'):
    clock_end = time.time()
    if printer:
        print(f'Runtime: {int((clock_end-start)//60)} min {int((clock_end-start)%60):2d} sec')
        return None
    else:
        return f' (...Runtime: {int((clock_end-start)//60)} min {int((clock_end-start)%60):2d} sec)'

    
