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

from utils.imports import print_runtime, vocab, batch_size, d_model, learning_rate, num_chars, encode, decode, plt, pylab, ls_pt


def load_val_data(device, world_size):
    """ len(val_data) = 500000
    """
    s0 = time.time()
    with open('dataset/val_data.json', 'r') as _f:
        val_data = json.load(_f)

    val_data = torch.tensor(val_data, dtype=torch.long, device='cpu')
    if device == 0:
        print(f'len(val_data):{len(val_data)}  {print_runtime(s0, False)}')

    return val_data


def clean_up(text, vocab):
    # replace out of vocab chars with ' '
    list_text = list(text)
    for i, c in enumerate(list_text):
        if c not in vocab:
            list_text[i] = ' '

    # idx: Remove the indices that are whitespaces and are followed by another white space.
    idx = [i for i in range(len(list_text)-1) if list_text[i] == list_text[i+1] == ' ']
    list_text = [list_text[i] for i in range(len(list_text)) if i not in set(idx)]

    text = ''.join(list_text)
    return text


def ptxt(num_chars):
    if num_chars < 1e6:
        txt = f'{num_chars*1e-3:3.2f} K'
    elif num_chars < 1e9:
        txt = f'{num_chars*1e-6:3.2f} M'
    elif num_chars < 1e12:
        txt = f'{num_chars*1e-9:3.2f} G'

    return txt


def load_google_corpus(device, idx_file):
    s0 = time.time()

    # fname = '/data/home/osafak/code/mygpt/dataset/news_tensors/europarl-v6.en_000.pt'
    fname = ls_pt[idx_file % len(ls_pt)]
    data = torch.load(fname)
    data = data.to(torch.long)

    if device == 0:
        print(f'load_google_corpus: idx_file:{idx_file} -- {idx_file/len(ls_pt)*100:.0f} % of trainset files')
        print(f'         {fname.split("/")[-1]} --  len(data):{len(data)/1e6:.2f} million tokens -- {print_runtime(s0, False)}')
    
    return data, idx_file + 1


def load_openwebtext_data():
    s0 = time.time()
    data_dir = os.path.join(os.getcwd(), 'dataset/openwebtext/')
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    return train_data, val_data


def get_grad_vector(model):
    list_grads = []
    dict_grads = dict()
    dict_weights = dict()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            list_grads.append(param.grad.view(-1,1))
            dict_grads[name] = param.grad
            dict_weights[name] = param
        else:
            continue
    grad_vector = torch.cat(list_grads)
    return grad_vector, dict_grads, dict_weights


def plotter(model, device, list_steps, list_losses, list_lr, list_ppl_val, list_steps_val, 
            list_losses_val, list_secs, start, savefig=True):
    """ list_secs:  Wall time for accrued single batch  
    """
    
    if device != 0:
        return
    
    s0 = time.time()
    
    step = len(list_losses)
    list_steps = np.array(list_steps)
    list_steps_val = np.array(list_steps_val)
    
    # define subplots layout 
    fig = plt.figure(figsize=(3.5 * 1.618 * 2, 3.5 * 3))
    spec = fig.add_gridspec(3, 2, height_ratios=[1.5, 2, 2])
    ax00 = fig.add_subplot(spec[0, :])
    ax10 = fig.add_subplot(spec[1, 0])
    ax11 = fig.add_subplot(spec[1, 1])
    ax20 = fig.add_subplot(spec[2, 0])
    ax21 = fig.add_subplot(spec[2, 1])

    ax00.set_axis_off()
    ax00.text(0, 0.9, model.module.specs, ha='left', va='top', family='monospace', size='smaller')
    ax00.text(0, 0.5, model.module.table, ha='left', va='top', family='monospace', size='smaller')

    ax10.semilogy(list_steps_val, list_losses_val, 'r-', alpha=.6, label=f'val {min(list_losses_val):.2f}')
    ax10.semilogy(list_steps, list_losses, 'k-', alpha=.6, label=f'train {min(list_losses):.2f}')
    ax10.set_title(f'Cross-Entropy Loss (step={step}) {print_runtime(start, False)} ')
    ax10.set_ylim(min(3, min(list_losses)-0.1), 11)
    
    ax11.plot(list_steps, list_lr, 'k', alpha=.5, label='learning_rate')
    ax11.set_ylim(0)

    ax20.plot(list_steps, list_secs, 'k.', alpha=.5, label='Wall time per step')
    ax20.set_ylabel('sec')

    ax21.semilogy(list_steps_val, list_ppl_val, 'k-', alpha=.6, label=f'PPL val\n{min(list_ppl_val):.2f}')
    [ax.set_xlabel('steps') for ax in [ax11, ax21]]
    [(ax.legend(), ax.set_xlim(0)) for ax in [ax10, ax11, ax20, ax21]]
    
    if savefig:
        pst = pytz.timezone('US/Pacific')
        delta = - datetime.timedelta(hours=8) + datetime.timedelta(minutes=20) + datetime.timedelta(seconds=42)
        dt = datetime.datetime.now(pst) + delta
        prefix = dt.isoformat().split('.')[0]
        prefix = prefix.replace('T', ' | ')
        plt.savefig(f'figures/loss_{step:05d}.png', bbox_inches='tight')
        print(f'Saved plot')
    else:
        plt.show()

    plt.close()








