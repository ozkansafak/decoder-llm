import os
import time
import json
import torch
import numpy as np
from pprint import pprint
from urllib.request import urlopen
from torch import nn
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from collections import Counter
from bs4 import BeautifulSoup
from prettytable import PrettyTable
from utils.imports import (
    print_runtime,
    vocab,
    num_chars,
    encode,
    decode,
    plt,
    pylab,
    ls_pt,
)


def clean_up(text: str, vocab: set) -> str:
    list_text = [c if c in vocab else ' ' for c in text]
    cleaned_text = ''.join(list_text)
    return cleaned_text


def ptxt(num_chars: int) -> str:
    if num_chars < 1e6:
        return f'{num_chars * 1e-3:3.2f} K'
    if num_chars < 1e9:
        return f'{num_chars * 1e-6:3.2f} M'
    if num_chars < 1e12:
        return f'{num_chars * 1e-9:3.2f} G'
    return ''


def load_google_corpus(device: int, idx_file: int) -> (torch.Tensor, int):
    s0 = time.time()
    fname = ls_pt[idx_file % len(ls_pt)]
    data = torch.load(fname).to(torch.long)

    if device == 0:
        print(f'Loaded Google Corpus: idx_file {idx_file} -- {idx_file / len(ls_pt) * 100:.0f}% of trainset files')
        print(f'{fname.split("/")[-1]} --  len(data): {len(data) / 1e6:.2f} million tokens -- {print_runtime(s0, False)}')

    return data, idx_file + 1


def load_openwebtext_data() -> (np.ndarray, np.ndarray):
    s0 = time.time()
    data_dir = os.path.join(os.getcwd(), 'dataset/openwebtext/')
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    return train_data, val_data


def get_grad_vector(model: nn.Module) -> torch.Tensor:
    list_grads = [param.grad.view(-1, 1).detach().cpu() for name, param in model.named_parameters() if param.requires_grad and param.grad is not None]
    grad_vector = torch.cat(list_grads)
    return grad_vector


def min2(inputs: list) -> int:
    return min(filter(None, inputs)) if any(inputs) else 0


def plotter(
    model: nn.Module,
    device: int,
    list_steps: list,
    list_losses: list,
    list_lr: list,
    list_ppl_val: list,
    list_steps_val: list,
    list_losses_val: list,
    list_secs: list,
    start: float,
    savefig: bool = True,
) -> None:
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

    ax10.semilogy(list_steps_val, list_losses_val, 'r-', alpha=.6, label=f'val {min2(list_losses_val):.2f}')
    ax10.semilogy(list_steps, list_losses, 'k-', alpha=.6, label=f'train {min2(list_losses):.2f}')
    ax10.set_title(f'Cross-Entropy Loss (step={step}) {print_runtime(start, False)} ')
    ax10.set_ylim(min(3, min2(list_losses)-0.1), 11)
    
    ax11.plot(list_steps, list_lr, 'k', alpha=.5)
    ax11.set_title('learning_rate')
    ax11.set_ylim(0)

    ax20.plot(list_steps, list_secs, 'k.', alpha=.5, label='Wall time per step')
    ax20.set_ylabel('sec')

    ax21.plot(list_steps_val, list_ppl_val, 'k-', alpha=.6, label=f'test perplexity\n{min2(list_ppl_val):.2f}')
    ax21.set_ylim(0, 90)

    [ax.set_xlabel('steps') for ax in [ax11, ax21]]
    [ax.legend() for ax in [ax10, ax20, ax21]]
    [ax.set_xlim(0) for ax in [ax10, ax20, ax11, ax21]]
    
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








