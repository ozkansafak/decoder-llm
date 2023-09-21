import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

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

    from utils.imports import vocab_size, learning_rate, num_chars, tokenizer, acc_batch_size
    from utils.helpers import load_val_data
    from utils.model import initialize_model, load_openwebtext_data, train, load_ckpt
    torch.manual_seed(device)
    # default values
    num_tokens_init = q_init = step_init = 0
    
    # instantiate model
    model, optimizer = initialize_model(vocab_size, device, learning_rate)
    model.to(device)
    
    # load a previous checkpoint
    PATH='/data/home/osafak/code/decoder-llm/saved_runs/304M_v2/models/chkpt_22000.pt'
    step_init = load_ckpt(device, model, optimizer, PATH)
    num_tokens_init = (step_init * 516096)
    q_init = num_tokens_init // acc_batch_size
    
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    # load train_data and val_data 
    train_data, val_data = load_openwebtext_data()

    # train loop
    train(device, model, optimizer, train_data, val_data, world_size, 
          step_init, num_tokens_init, q_init)

    destroy_process_group()


if __name__ == '__main__':
    import argparse, time
    from utils.imports import world_size, tokenizer
    print(f"tokenizer: {tokenizer}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    parser = argparse.ArgumentParser(description='simple distributed training job')
    args = parser.parse_args()

    for file in glob.glob(os.path.join('figures/loss*.png')):
        os.remove(file)

    mp.spawn(main, args=(world_size,), nprocs=world_size)

    
    
    
    
    
    
    
    