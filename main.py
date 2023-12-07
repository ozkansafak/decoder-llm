# Standard Libraries
import os
import glob
import argparse
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Local imports
from utils.imports import (
    vocab_size,
    world_size,
    config
)
from utils.model import (
    initialize_model,
    load_openwebtext_dataset,
    train,
    load_ckpt
)

# Distributed Data Parallel Setup
def ddp_setup(device: torch.device, world_size: int) -> None:
    """
    Setup DDP
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="NCCL", rank=device, world_size=world_size)
    torch.cuda.set_device(device)


# Main Training Loop across 8 GPUs
def main(device: torch.device, world_size: int) -> None:
    """
    Main training loop to be run across multiple GPUs.
    """
    # Initialize DDP and set manual seed
    ddp_setup(device, world_size)
    torch.manual_seed(device)
    
    # Initialize model and optimizer
    model, optimizer = initialize_model(vocab_size, device, config['learning_rate'])
    model.to(device)
    
    # Load checkpoint here when you want to.
    PATH = '/data/home/osafak/code/decoder-llm/saved_runs/304M_v2/models/chkpt_22000.pt'
    step_init = load_ckpt(device, model, optimizer, PATH)
    num_tokens_init = step_init * 516096
    q_init = num_tokens_init // config['acc_batch_size']
    
    # Convert model to DDP
    model = DDP(model, device_ids=[device], find_unused_parameters=True)

    # Load data and start training loop
    train_data, val_data = load_openwebtext_dataset()
    train(device, model, optimizer, train_data, val_data, world_size, step_init, num_tokens_init, q_init)
    
    # Destroy process group post-training
    destroy_process_group()


if __name__ == '__main__':
    # Print tokenizer and CUDA_VISIBLE_DEVICES information
    print(f"tokenizer: {config['tokenizer']}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Remove existing loss figure files to start from a clean slate
    for file in glob.glob('figures/loss*.png'):
        os.remove(file)
    
    # Input argument parsing
    parser = argparse.ArgumentParser(description='Simple distributed training job')
    args = parser.parse_args()
    
    # Spawn main process for each device
    mp.spawn(main, args=(world_size,), nprocs=world_size)
