# Import standard libraries
import os
import time
import datetime
import pytz

# Import scientific computing libraries
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import PyTorch neural network module
from torch import nn

# Import custom utilities
from utils.imports import print_runtime, ls_pt


def clean_text(text: str, vocab: set) -> str:
    cleaned_text = ''.join([char if char in vocab else ' ' for char in text])
    return cleaned_text

def pretty_print_number(num_chars: int) -> str:
    if num_chars < 1e6:
        return f'{num_chars * 1e-3:3.2f} K'
    elif num_chars < 1e9:
        return f'{num_chars * 1e-6:3.2f} M'
    elif num_chars < 1e12:
        return f'{num_chars * 1e-9:3.2f} G'
    return ''


def load_google_corpus(device: int, idx: int) -> (torch.Tensor, int):
    start_time = time.time()
    filename = ls_pt[idx % len(ls_pt)]
    data = torch.load(filename).long()
    
    if device == 0:
        print(f'Loaded Google Corpus: file index {idx} -- {idx / len(ls_pt) * 100:.0f}% completed')
        print(f'{filename.split("/")[-1]} --  Data Length: {len(data) / 1e6:.2f} million tokens -- {print_runtime(start_time, False)}')

    return data, idx + 1


def load_openwebtext_dataset() -> (np.ndarray, np.ndarray):
    start_time = time.time()
    data_dir = os.path.join(os.getcwd(), 'dataset/openwebtext/')
    training_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    validation_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    return training_data, validation_data



def get_grad_vector(model: nn.Module) -> torch.Tensor:
    list_grads = [param.grad.view(-1, 1).detach().cpu() for name, param in model.named_parameters() if param.requires_grad and param.grad is not None]
    grad_vector = torch.cat(list_grads)
    return grad_vector


def find_min_non_zero(inputs: list) -> int:
    return min(x for x in inputs if x is not None) if any(inputs) else 0


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








