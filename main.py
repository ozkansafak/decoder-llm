# Standard and Third-Party Libraries
import os
import sys
import glob
import json
import time
import pickle
import random
import argparse
import datetime
from collections import Counter
from urllib.request import urlopen

# External Packages
import ipdb
import re
import pytz
import numpy as np
import torch
import torch.multiprocessing as mp
import unidecode
import unicodedata
from bs4 import BeautifulSoup
from pprint import pprint
from prettytable import PrettyTable
from torch import nn
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken

# Local Utilities and Configs
from utils.imports import vocab_size, learning_rate, num_chars, tokenizer, acc_batch_size, world_size
from utils.helpers import load_val_data
from utils.model import initialize_model, load_openwebtext_data, train, load_ckpt


# Distributed Data Parallel Setup
def ddp_setup(device: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="NCCL", rank=device, world_size=world_size)
    torch.cuda.set_device(device)


# Main Training Loop across 8 GPUs
def main(device, world_size):
    ddp_setup(device, world_size)
    torch.manual_seed(device)
    
    model, optimizer = initialize_model(vocab_size, device, learning_rate)
    model.to(device)
    
    PATH = '/data/home/osafak/code/decoder-llm/saved_runs/304M_v2/models/chkpt_22000.pt'
    step_init = load_ckpt(device, model, optimizer, PATH)
    num_tokens_init = step_init * 516096
    q_init = num_tokens_init // acc_batch_size
    
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    train_data, val_data = load_openwebtext_data()
    train(device, model, optimizer, train_data, val_data, world_size, 
          step_init, num_tokens_init, q_init)
    
    destroy_process_group()


if __name__ == '__main__':
    print(f"tokenizer: {tokenizer}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    for file in glob.glob('figures/loss*.png'):
        os.remove(file)
    
    parser = argparse.ArgumentParser(description='Simple distributed training job')
    args = parser.parse_args()
    
    mp.spawn(main, args=(world_size,), nprocs=world_size)
