from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import ipdb, re, time, sys, os, pickle, glob, json, random, unidecode, unicodedata
from IPython.display import clear_output
from collections import Counter
from urllib.request import urlopen
from bs4 import BeautifulSoup

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from pprint import pprint
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch
from torch import nn
from torch.nn import functional as F

#from transformers import AutoTokenizer, AutoModel
import tiktoken


pylab.rcParams.update({'legend.fontsize': 'medium',
                       'font.size'      : 14,
                       'figure.figsize' : (18, 4),
                       'axes.labelsize' : 'medium',
                       'axes.titlesize' : 'medium',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'medium',
                       'ytick.labelsize': 'medium'})


def print_runtime(start, printer=True, end='\n'):
    clock_end = time.time()
    if printer:
        print(f'Runtime: {int((clock_end-start)//60)} min {int((clock_end-start)%60):2d} sec')
        return None
    else:
        return f' (...Runtime: {int((clock_end-start)//60)} min {int((clock_end-start)%60):2d} sec)'

    
