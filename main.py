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
import tiktoken 

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(device: int, world_size: int):
    """
    Args: 
        device: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="NCCL", rank=device, world_size=world_size)
    torch.cuda.set_device(device)


def main(device, world_size):
    ddp_setup(device, world_size)
    
    from utils.imports import print_runtime, count_parameters, d_head, vocab_size, vocab, new_links, visited_urls, batch_size, d_model, n_heads, n_layer, block_size, learning_rate, dropout, max_iters, eval_steps, num_chars, add 

    from utils.helpers import load_val_data, extract_single_url, get_links, shave, decompose_divs, plotter, clean_up, ptxt, crawl_wiki_data

    from utils.model import DecoderModel, generate_text, estimate_loss, load_train_objs, train
    
    torch.manual_seed(device)

    # instantiate model
    model, optimizer = load_train_objs(vocab_size, device, learning_rate)
    model.to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True) 
    
    # load val_data by crawling the list of wiki pages in "dataset/val_wiki.json"
    val_data, val_urls = load_val_data(device, num_pages=30)
    list_num_tokens, list_num_tokens_val, list_losses, list_losses_val = [], [], [], []

    # train loop
    train(device, model, optimizer, num_chars, val_data, 
          list_num_tokens, list_losses, list_num_tokens_val, list_losses_val, eval_steps)

    destroy_process_group()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    args = parser.parse_args()

    for file in glob.glob(os.path.join('figures/*.png')):
        os.remove(file)

    world_size = torch.cuda.device_count()
    print(f'world_size:{world_size}')
    mp.spawn(main, args=(world_size,), nprocs=world_size)

    """
        NOTES:
         -  plotter variable 'list_num_tokens', 'list_losses'..  how are they being reduced?
         -  Only Master GPU should do the plotting. 
         -  'visited_urls' should be gathered into 'all_visited_urls'
         -  URLs have been stored in a json file. They should be crawled in CPU then distributed to GPUs.
         -  val_data shoild be wrapped in torch dataLoader class just as train_data is.
         
    """

    
    
    
    
    
    
    
    