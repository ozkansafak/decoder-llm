# Standard Libraries
from typing import List, Dict, Any
import yaml
import time
import glob
import torch
from prettytable import PrettyTable
import matplotlib.pylab as pylab
import tiktoken 
from tiktoken_ext.openai_public import ENCODING_CONSTRUCTORS

# Matplotlib Configuration
pylab.rcParams.update({'legend.fontsize': 'small',
                       'font.size'      : 12,
                       'figure.figsize' : (9, 3.5),
                       'axes.labelsize' : 'small',
                       'axes.titlesize' : 'small',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'small',
                       'ytick.labelsize': 'small'})

# Global Configuration
CONFIG_FILE_PATH = "config.yml"
DATA_DIRECTORY = "/data/home/osafak/code/mygpt/dataset/news_tensors"
DATA_PATH = f"{DATA_DIRECTORY}/*.pt"

str_vocab = '\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
vocab = set(str_vocab)

# Load Configuration and Hyperparameters
def load_config(config_file: str) -> Dict[str, Any]:
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: Configuration file not found.")
        exit(1)

config = load_config(CONFIG_FILE_PATH)
config['d_head'] = int(config['d_model'] / config['n_heads'])

# Validate Configurations
assert config['tokenizer'] in ['gpt2', 'character']
assert (config['d_model'] / config['n_heads']) % 1 == 0

num_chars = 0 

# Compute Batch Size and Handle Inconsistencies
world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1 # 7
config['batch_size'] = config['batch_size_gpu'] * world_size  # 42
if config['batch_size'] % world_size > 0:
    print(f"===> batch_size not a multiple of world_size: (batch_size % world_size = {config['batch_size'] % world_size}) "+
          f"batch_size will be clipped to {(config['batch_size'] // world_size) * world_size}")
    config['batch_size'] = (config['batch_size'] // world_size) * world_size

# this is batch_size (B) per gpu.
config['batch_jump'] = (config['context_length'] * config['batch_size'])  # number of tokens ingested in each batch
config['acc_batch_size'] = (config['acc_batch_size'] // config['batch_jump']) * config['batch_jump']  
config['num_chunked_batches'] = int(config['acc_batch_size'] / config['batch_jump']) # number of loss.backward() made before doing an optim.step()

_directory = "/data/home/osafak/code/mygpt/dataset/news_tensors"
_PATH = f"{_directory}/*.pt"
ls_pt = glob.glob(_PATH)
ls_pt.sort(key=lambda x: x.split("/")[-1])

def print_trainset_details() -> None:
    """
        europarl-v6.en_000.pt        -- 59.34 million tokens
        news-commentary-v6.en_000.pt -- 4.81 million tokens  (used as validation set)
        news.2007.en.shuffled_000.pt -- 392.89 million tokens
        news.2008.en.shuffled_000.pt -- 900.00 million tokens
        news.2008.en.shuffled_001.pt -- 68.16 million tokens
        news.2009.en.shuffled_000.pt -- 900.00 million tokens
        news.2009.en.shuffled_001.pt -- 275.56 million tokens
        news.2010.en.shuffled_000.pt -- 458.62 million tokens
        news.2011.en.shuffled_000.pt -- 63.05 million tokens
    """

    if torch.distributed.get_rank() == 0:
        print('Trainset:')
        for fname in ls_pt:
            data = torch.load(fname)
            print(f"{fname.split('/')[-1]:<28} -- {len(data) / 1e6:.2f} million tokens")


def print_runtime(start: float, printer: bool = True) -> None:
    end = time.time()
    if printer:
        print(f'Runtime: {int((end - start) // 60)} min {int((end - start) % 60):2d} sec')
    else:
        return f'({int((end - start) // 60)} min {int((end - start) % 60):2d} sec)'


def count_parameters(model: torch.nn.Module) -> int:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param_count = parameter.numel()
        table.add_row([name, param_count])
        total_params += param_count
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if config['tokenizer'] == 'gpt2':
    encFunc = ENCODING_CONSTRUCTORS['gpt2']
    encDict = encFunc() 
    enc = tiktoken.Encoding(encDict['name'], pat_str=encDict['pat_str'], mergeable_ranks=encDict['mergeable_ranks'], 
                            special_tokens=encDict['special_tokens' ])
    vocab_size = enc.n_vocab
    encode = enc.encode
    decode = enc.decode
elif config['tokenizer'] == 'character':
    vocab_size = len(vocab)
    stoi = {s:i for i,s in enumerate(str_vocab)}
    itos = {i:s for i,s in enumerate(str_vocab)}
    def encode(str_input):
        return [stoi[c] for c in str_input]
    def decode(list_idx):
        return ''.join([itos[i] for i in list_idx])
