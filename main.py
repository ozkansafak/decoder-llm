# CELL 1
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

def ddp_setup(rank: int, world_size: int):
    """
    Args: 
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(pend, device):
    from utils.imports import print_runtime, count_parameters, d_head, vocab_size, vocab, list_vocab, new_links, visited_urls, batch_size, d_model, n_heads, n_layer, block_size, learning_rate, dropout, max_iters, eval_steps, num_chars, add

    from utils.helpers import load_val_data, extract_single_url, get_links, shave, decompose_divs, plotter, clean_up, ptxt, crawl_wiki_data

    from utils.model import DecoderModel, generate_text, estimate_loss, get_batch, load_train_objs, train

    # instantiate model
    model, optimizer = load_train_objs(vocab_size, device, learning_rate)

    # load val_data by crawling the list of wiki pages in "dataset/val_wiki.json"
    val_data, val_urls = load_val_data(num_pages=10)
    list_num_tokens, list_num_tokens_eval, list_losses, list_losses_eval = [], [], [], {'train':[], 'val':[]}

    # train loop
    train(model, optimizer, device, num_chars, val_data, 
          list_num_tokens, list_losses, list_num_tokens_eval, list_losses_eval, eval_steps, pend=pend)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--pend', default='\n', type=str, help='print(..., end=pend) end token')
    parser.add_argument('--device', default='6', type=str, help='single GPU device code [0,..7]')
    args = parser.parse_args()
    args.device = f'cuda:{args.device}'

    main(args.pend, args.device)




    
    
    
    
    
    
    
    