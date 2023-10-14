from typing import Tuple, List, Optional

import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from prettytable import PrettyTable

from torch.utils.data.distributed import DistributedSampler

from utils.imports import (
    print_runtime, 
    config, 
    batch_jump, 
    encode, 
    decode, 
    world_size, 
    acc_batch_size, 
    num_chunked_batches
)

from utils.helpers import plotter, get_grad_vector

# Seed for reproducibility
torch.manual_seed(1337)


class SingleHead(nn.Module):
    """ Single self-attention head """
    def __init__(self):
        super().__init__()
        self.W_K = nn.Linear(config['d_model'], config['d_head'], bias=False)
        self.W_Q = nn.Linear(config['d_model'], config['d_head'], bias=False)
        self.W_V = nn.Linear(config['d_model'], config['d_head'], bias=False)
        """ torch.tril: retains lower triangular part of 2-D matrix, and sets the other elements to 0. 
            Here, we register_buffer self.tril so the gradients of the values of the mask won't be computed.
               >  If you have parameters in your model, which should be saved and restored in the state_dict,
               >  but not trained by the optimizer, you should register them as buffers.
        """
        
        # Dropout layer to regularization
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x: torch.Tensor, eos_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # (batch_size_gpu, context_length, d_model)
        # idx and targets are both (B,T) tensor of integers
        K = self.W_K(x) # (B, T, d_head) 
        Q = self.W_Q(x) # (B, T, d_head)
        V = self.W_V(x) # (B, T, T)

        # Computer attention scores ("affinities")
        Attn = torch.einsum('bik,bjk->bij', Q, K) / np.sqrt(config['d_head'])
        Attn = Attn.masked_fill(eos_mask[:B, :T, :T] == 0, float('-inf'))  # (B, T, T)
        Attn = F.softmax(Attn, dim=-1) # (B, T, T)
        Attn = self.dropout(Attn) # randomly prevent nodes from communicating

        # Perform weighted aggregation of the values 
        out = torch.einsum('bij,bjk->bik', Attn, V) # (B, T, d_head)

        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel 
        num_heads: each self-attention head is a communication channel for the tokens in sequence
        d_head: dimension of k,q,v vectors
    """
    def __init__(self, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead() for _ in range(num_heads)]) # type: nn.ModuleList
        self.proj = nn.Linear(config['d_model'], config['d_model']) # type: nn.Linear
        self.dropout = nn.Dropout(config['dropout']) # type: nn.Dropout


    def forward(self, x: torch.Tensor, eos_mask: torch.Tensor) -> torch.Tensor:
        # Concatenate over the channel dimension, then pass it through a linear layer to project.
        out = torch.cat([h(x, eos_mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    """ Simple multi-layer perceptron """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            # hidden layer size is 4x the input layer
            nn.Linear(d_model, 4 * d_model),  
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config['dropout']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication (MultiHeadAttention) followed by computation (MLP) """

    def __init__(self, d_model: int, n_heads: int):
        # d_model: embedding dimension, n_heads: the number of heads we'd like
        super().__init__() 
        self.attention = MultiHeadAttention(n_heads)
        self.mlp = MLP(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, ins: tuple) -> tuple:
        x, eos_mask = ins

        # Residual connections. Fork off, apply a function and join with the main stream with an addition. 
        # SAFAK-5
        #x = self.ln1(x)
        #x = x + self.attention(x, eos_mask)
        #x = self.ln2(x)
        #x = x + self.mlp(x)
        
        # GPT-2 and karpathy's nano-gpt
        # Apply attention and MLP layers
        x = x + self.attention(self.ln1(x), eos_mask)
        x = x + self.mlp(self.ln2(x))
        
        return x, eos_mask


class DecoderModel(nn.Module):
    def __init__(self, vocab_size: int, device: torch.device, config: dict):
        super().__init__()
        # super(PositionalEncoding, self).__init__()
        # each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, config['d_model'])
        self.position_embedding_table = PositionalEncoding(config['context_length'], config['d_model'], config['dropout'])  # (1,T,C)
        
        self.blocks = nn.Sequential(*[Block(config['d_model'], config['n_heads']) for _ in range(config['n_layers'])])
        self.ln_f = nn.LayerNorm(config['d_model']) # final layer norm
        self.lm_head = nn.Linear(config['d_model'], vocab_size) # 
        self.device = device
        self.num_params = None
        self.footprint = None
        self.table = None
        self.specs = None
        self.print_hyperparams_to_stdout()
        self.register_buffer('eos_mask_template', torch.tril(torch.ones(config['context_length'], config['context_length'])))

    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # inputs and targets are both shape (B,T) tensor of integers
        B, T = inputs.shape
        
        # compute the eos_mask
        eos_token_id = 50256 # end of token index for in OpenWebText dataset and gpt2 tokenizer
        idx = (inputs == eos_token_id).nonzero(as_tuple=True)
        eos_mask = self.eos_mask_template.repeat(B,1,1)
        for b, t in zip(*idx): # T
            eos_mask[b, t:, :t] = 0 # this needs to be [b, t:, :(t+1)]
        
        token_embeddings = self.token_embedding_table(inputs) # (B,T,d_model)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=inputs.device)) # (1,T,C)-PositionalEncoding
        x = token_embeddings + position_embeddings # (B,T,d_model)
        x, _ = self.blocks((x, eos_mask)) # (B,T,d_model)
        x = self.ln_f(x) # todo: This line was left out. Train again with GPT-2 style Blocks and with ln_f.
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def print_hyperparams_to_stdout(self) -> int:
        num_params = 0
        for name, item in self.named_parameters():
            if len(item.shape) == 1:
                m = item.shape[0]
                num_params += m
            else:
                m, n = item.shape
                num_params += m * n

        if self.device == 0:
            if num_params < 1e3:
                str_num_params= (f"num_params: {num_params}")
            elif num_params < 1e6:
                str_num_params= (f"num_params: {int(num_params * 1e-3)} K")
            elif num_params < 1e9:
                str_num_params= (f"num_params: {int(num_params * 1e-6)} M")
            elif num_params < 1e12:
                str_num_params= (f"num_params: {(num_params * 1e-9):.3f} B")
            print(str_num_params)

            self.num_params = num_params
            self.footprint = num_params * next(self.parameters()).element_size()
            table = PrettyTable(['d_model', 'n_layers', 'n_heads', 'd_head', 'context_length', 'batch_size', 
                                 'acc_batch_size', 'learning_rate'])
            table.add_row([config['d_model'], config['n_layers'], config['n_heads'], config['d_head'], config['context_length'], 
                           f"{config['batch_size']}\n{config['batch_size_gpu']}/GPU", acc_batch_size, f"{config['learning_rate']:.2e}"])

            self.table = table
            self.specs = str_num_params +\
                         f"\ntokenizer:{config['tokenizer']}\nCUDA_VISIBLE_DEVICES:{os.environ['CUDA_VISIBLE_DEVICES']}"+\
                         f"\nworld_size:{world_size}"
            print(table)

        return num_params


class PositionalEncoding(nn.Module):  #@save
    """ Ref: https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html """
    def __init__(self, context_length: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create a long enough P
        pe = torch.zeros((1, context_length, d_model))
        div_term = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        x = torch.arange(context_length, dtype=torch.float32).reshape(-1, 1) / div_term
        pe[:, :, 0::2] = torch.sin(x)
        pe[:, :, 1::2] = torch.cos(x)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (T)
        x = x.unsqueeze(1).unsqueeze(0) # (1, T, 1)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False) # :x.size(1) is for when context_length < T 
        return self.dropout(x)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, context_length: int) -> None:
        if len(data) % context_length == 0:
            #  If no target token left for the last batch. 
            print(f'\n===> MyDataset: len(data) % context_length == 0')
            data = torch.cat(( data, torch.tensor(encode('.'), dtype=torch.long) ))

        total_batches = len(data) // context_length

        data2 = data[:total_batches * context_length]
        data2 = data2.view(total_batches, context_length)

        targets = data[1:total_batches*context_length+1]
        targets2 = targets.view(total_batches, context_length)

        self.data = data2
        self.targets = targets2

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


@torch.no_grad()
def perplexity(model: torch.nn.Module, 
               device: torch.device, 
               data: np.ndarray, 
               list_steps_val: List[int], 
               step: int, 
               list_losses_val: List[torch.Tensor], 
               list_ppl_val: List[torch.Tensor]) -> None:
    
    s0 = time.time()
    model.eval()
    
    ctr = ppl = loss_avgd = 0
    for q in range(min(4, len(data) // acc_batch_size)):
        # for OpenWebText val_data.bin there are 8 or 9 0.5M-token batches.
        q0 = q * acc_batch_size
        q1 = (q+1) * acc_batch_size + 1
        data_step = torch.from_numpy(data[q0:q1].astype(np.int64))
        mytestset = MyDataset(data_step, context_length=config['context_length'])
        test_loader = torch.utils.data.DataLoader(dataset=mytestset,
                                                  batch_size=(num_chunked_batches*config['batch_size']),
                                                  shuffle=False,
                                                  sampler=DistributedSampler(mytestset))

        for batch_no, (xb_big, yb_big) in enumerate(test_loader):
            for i in range(num_chunked_batches):
                xb = xb_big[i*config['batch_size_gpu'] : (i+1)*config['batch_size_gpu']]  # each device ==> (4, 512) 
                yb = yb_big[i*config['batch_size_gpu'] : (i+1)*config['batch_size_gpu']]

                logits, loss = model(xb, yb) # evaluate the loss
                ctr += 1
                ppl += torch.exp(loss)
                loss_avgd += loss

    ppl /= ctr
    loss_avgd /= ctr
    torch.distributed.all_reduce(ppl)
    torch.distributed.all_reduce(loss_avgd)
    ppl /= world_size
    loss_avgd /= world_size
    ppl = ppl.detach().to('cpu')
    loss_avgd = loss_avgd.detach().to('cpu')

    if is_main_process():
        print(f'=> Testset step:{step} -- ppl:{ppl:.2f} -- loss:{loss_avgd:.2f}  ctr:{ctr} q:{q} -- {print_runtime(s0, False)}')

    list_losses_val.append(loss_avgd)
    list_ppl_val.append(ppl)
    list_steps_val.append(step)


def generate_text(model: nn.Module, device: torch.device, step: Optional[int] = None, ppl: Optional[float] = None) -> None:
    if not is_main_process():
        return

    model.eval()
    s0 = time.time()
    seed_text = 'Scientists recently discovered that the tap water in our homes can be'    
    seed_tokens = torch.as_tensor(encode(seed_text), device=device, dtype=torch.long).view(1,-1)
    out_tokens = generate_tokens(model, device, idx=seed_tokens).tolist()
    output = f''.join(decode(out_tokens))
    print(f'\n===> In generate_text(): step:{step} -- ppl:{ppl:.2f} -- {print_runtime(s0, False)}')
    print(f'===> seed_text  : {seed_text}')
    print(f'===> decoder-llm: {output.split(seed_text)[1]}')
    print('---' *30 + '\n')
    model.train()


@torch.no_grad()
def generate_tokens(model: nn.Module, device: torch.device, idx: torch.Tensor, temperature: float = .5, max_new_tokens: int = 200) -> torch.Tensor:
    # idx is (B, T) array of indices in the current context
    
    for _ in range(max_new_tokens):
        # get prediction 
        # crop idx to the last context_length tokens
        logits, _ = model(idx[:, -context_length:]) # logits.shape: (1, T, n_vocab) 

        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        
        # temperature scaling
        logits = logits / temperature  

        # get softmax probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)

        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # idx:(1, T+1), idx_next:(1, 1)

    return idx[0]


class WarmupCosineAnnealing(torch.optim.lr_scheduler.LambdaLR):
    # https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    """ Linearly increases learning rate from 0.1 to 1 over `x0` training steps.
        Decreases learning rate from 1. to 0.1 over the next `x1 - x0` steps following a b/(x+a) hyperbolic curve.
        Stays constant at 0.1 after x1 steps.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, x0: float, x1: float, last_epoch: int = -1):
        self.x0 = x0 / acc_batch_size
        self.x1 = x1 / acc_batch_size
        super(WarmupCosineAnnealing, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, x: float) -> float:
        # .step() invokes this function through LambdaLR
        if x < self.x0:
            # linear warm up stage
            y = .9 * x / max(1, self.x0) + 0.1 
        elif self.x0 <= x < self.x1:
            # cosine annealing stage
            T = (self.x1 - self.x0) * 2
            A = .5 * .9
            y = A * np.cos(2 * np.pi * (x - self.x0) / T) + 0.55    
        elif self.x1 <= x:
            # constant 10% learning_rate
            y = .1

        return y


class WarmupInvXDecay(torch.optim.lr_scheduler.LambdaLR):
    # https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    """ Linearly increases learning rate from 0.1 to 1 over `x0` training steps.
        Decreases learning rate from 1. to 0.1 over the next `x1 - x0` steps following the y = b/(x+a)**0.5  curve.
        Stays constant at 0.1 after x1 steps.
        
        y = b / (x + a) ** 0.5
        y(x=x0) = 1
        y(x=x1) = p
    """

    def __init__(self, optimizer: torch.optim.Optimizer, x0: float, x1: float, p: float = 0.1, last_epoch: int = -1):
        # x0, and x1 are in "number of tokens"
        # self.x0 and self.x12 are scaled to step size.
        self.x0 = x0 / acc_batch_size
        self.x1 = x1 / acc_batch_size
        self.p = p
        self.a = 1 / (1 - self.p**2) * (self.p**2 * self.x1 - self.x0)
        self.b = np.sqrt(self.x0 + self.a)
        super(WarmupInvXDecay, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, x: float) -> float:
        # .step() invokes this function through LambdaLR. 
        # Each invokation of .step() increments self._step_count by 1.
        
        if x < self.x0:
            # linear warm up stage
            y = .9 * (x / self.x0) + 0.1
            
        elif self.x0 <= x <= self.x1:
            # inverse decay stage
            y = self.b / (x + self.a) ** 0.5
            
        elif self.x1 <= x:
            # constant stage
            y = self.p
            
        return y


def initialize_model(vocab_size: int, device: torch.device, learning_rate: float) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    # instantiate model
    model = DecoderModel(vocab_size, device)

    # Create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, lr=learning_rate)  
    
    optimizer.zero_grad(set_to_none=False)
    return model, optimizer


def load_shakespeare(num_chars: int = 0, filename: str = 'dataset/tiny_shakespeare.txt') -> Tuple[str, float]:
    """ 338,025 tokens
    """
    with open(filename, 'r', encoding='utf-8') as f:
        train_data = f.read()

    return train_data, num_chars + len(train_data) * 0.9


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_ddp_model(model: torch.nn.Module, device: torch.device, step: int, dname: str = 'models') -> None:
    s0 = time.time() 

    PATH = f'{dname}/chkpt_{step:05d}.pt'
    model.eval()

    if is_main_process():
        torch.save(model.state_dict(), PATH)
        print(f'Saved model to {PATH}  {print_runtime(s0, False)}')
        
    torch.distributed.barrier()
    return


def load_model(model: torch.nn.Module, PATH: str) -> torch.nn.Module:
    """
    Load the model from a given PATH.

    Parameters:
    model (torch.nn.Module): Model into which the parameters should be loaded.
    PATH (str): Filepath to the saved model.

    Returns:
    torch.nn.Module: Model with loaded parameters.
    """
    model.load_state_dict(torch.load(PATH))
    model.eval()


def save_ckpt(device: torch.device, model: nn.Module, optimizer: nn.Optimizer, step: int) -> None:
    """
    Save the model and optimizer state to a checkpoint file.
    
    Parameters:
    - device (torch.device): The device on which the model is running.
    - model (nn.Module): The PyTorch model object whose parameters will be saved.
    - optimizer (nn.Optimizer): The optimizer whose state will be saved.
    - step (int): The current training step (used for naming the checkpoint file).
    """
    s0 = time.time()

    PATH = f'models/chkpt_{step:05d}.pt'
    model.eval()

    if is_main_process():
        s0 = time.time() 
        state = {'model': model.module.state_dict(),
                 'optimizer': optimizer.state_dict(),
                }
        torch.save(state, PATH)
        print(f'Checkpoint saved at {PATH}  {print_runtime(s0, False)}') 
    torch.distributed.barrier()

def load_ckpt(device: torch.device, model: nn.Module, optimizer: nn.Optimizer, PATH: str) -> int:
    """
    Load the model and optimizer state from a checkpoint file.
    
    Parameters:
    - device (torch.device): The device on which the model is running.
    - model (nn.Module): The PyTorch model object into which the saved parameters will be loaded.
    - optimizer (nn.Optimizer): The optimizer into which the saved state will be loaded.
    - PATH (str): The file path of the saved model checkpoint.
    
    Returns:
    - int: The training step at which the model was saved.
    """
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    #model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    step_init = int(PATH.split('/')[-1].split('.pt')[0].split('_')[-1])
    
    if is_main_process():
        print(f'model checkpoint loaded step_init:{step_init} -- PATH:{PATH}')
    
    return step_init


def load_ddp_model_to_single_device(model:nn.Module, PATH: str) -> None:
    """
    Load a Distributed Data Parallel (DDP) model checkpoint to a single device.
    
    Parameters:
    - model (Module): The PyTorch model object into which the saved parameters will be loaded.
    - PATH (str): The file path of the saved model checkpoint.
    """

    from collections import OrderedDict

    state_dict = torch.load(PATH)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        name = key[7:] # remove `module.`
        new_state_dict[name] = value

    # Here, model should be a model.DecoderModel object 
    model.load_state_dict(new_state_dict)


def train(device: torch.device, model: torch.nn.Module, optimizer: Optimizer, 
          train_data: np.ndarray, val_data: np.ndarray, world_size: int, 
          step_init: int = 0, num_tokens_init: int = 0, q_init: int = 0) -> None:
    """
    Train a machine learning model.

    Parameters:
        device (torch.device): The device where the model will be trained.
        model (torch.nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer to use for training.
        train_data (np.ndarray): The training data.
        val_data (np.ndarray): The validation data.
        world_size (int): The size of the world for distributed training.
        step_init (int, optional): Initial step number. Default is 0.
        num_tokens_init (int, optional): Initial number of tokens. Default is 0.
        q_init (int, optional): Initial batch counter. Default is 0.
    
    Returns:
        None
    """
    
    start = time.time()
    if is_main_process():
        print(f"world_size: {world_size}")
        print(f"batch_size: {config['batch_size']}")
        print(f"context_length: {config['context_length']}")
        print(f"batch_jump: {batch_jump}")
        print(f"x0: {config['x0']/1e9:.3f}e9")
        print(f"x1: {int(config['x1']/1e9)}e9")
        print(f"clip: {config['clip']:.2f}")
        print(f"acc_batch_size: {acc_batch_size}")
        print(f"num_chunked_batches: {num_chunked_batches}")
        print(f"step_init: {step_init}")
        print(f"num_tokens_init: {num_tokens_init}")
        print(f"q_init: {q_init}")
        
    
    (list_steps, list_losses, list_steps_val, list_losses_val) = [[] for _ in range(4)]
    (list_lr, list_secs, list_ppl_val, list_grads) = [[] for _ in range(4)]
    num_tokens = sample_no = step = 0
    n = len(train_data)
    lr_scheduler = WarmupCosineAnnealing(optimizer, x0=config['x0'], x1=config['x1'])

    perplexity(model, device, val_data, list_steps_val, step, list_losses_val, list_ppl_val)

    if step_init > 0:
        num_tokens = num_tokens_init
        for step in range(1, step_init+1):
            lr_scheduler.step()
            list_lr.append(optimizer.param_groups[-1]['lr'])
            list_steps.append(step)
            list_losses.append(None)
            list_secs.append(None)
            list_steps_val[0] = list_ppl_val[0] = list_losses_val[0] = None
            if is_main_process() and step % 1000 == 0:
                print(f'Restarting from checkpoint. Fastforward batches step:{step} -- num_tokens:{num_tokens:.2e}')

    # train-loop
    for epoch in range(20):
        if is_main_process():
            print(f'\n {"---"*30}\n\nepoch:{epoch} -- num_chunked_batches:{num_chunked_batches} -- step:{step}')
            print(f'shaved number of tokens in train_data: {n -  (n // acc_batch_size)* acc_batch_size}', end='')
            print(f' -- {((n // acc_batch_size)* acc_batch_size) / n * 100 : .2f} % train_data retained')

        for q in range(0, n // acc_batch_size):
            if step_init > 0 and epoch == 0 and q < q_init:
                continue
            elif step_init > 0 and epoch == 0 and q == q_init and is_main_process():
                print(f'Restarting from checkpoint. Fastforward batches step:{step} -- q:{q} -- num_tokens:{num_tokens:.2e}')

            q0 = q * acc_batch_size
            q1 = (q+1) * acc_batch_size + 1
            data_step = torch.from_numpy(train_data[q0:q1].astype(np.int64))

            mytrainset = MyDataset(data_step, context_length=config['context_length'])
            train_loader = torch.utils.data.DataLoader(dataset=mytrainset,
                                                       batch_size=(num_chunked_batches*[config'batch_size']),
                                                       shuffle=False,
                                                       sampler=DistributedSampler(mytrainset))

            s0 = time.time()
            for batch_no, (xb_big, yb_big) in enumerate(train_loader):
                model.train()
                xb_big = xb_big.to(device)  # each device ==> (34 * 4, 512)
                yb_big = yb_big.to(device)  
                total_size = 0
                for i in range(num_chunked_batches):
                    xb = xb_big[i*config['batch_size_gpu'] : (i+1)*config['batch_size_gpu']]  # each device ==> (4, 512) 
                    yb = yb_big[i*config['batch_size_gpu'] : (i+1)*config['batch_size_gpu']]
                    num_tokens += len(xb) * config['context_length'] * world_size
                    total_size += len(xb) * config['context_length'] * world_size
                    if is_main_process():
                        print('.', end='')

                    logits, loss = model(xb, yb) # evaluate the loss
                    loss.backward() # adds to the gradients

                if total_size != acc_batch_size:
                    if is_main_process():
                        print(f'total_size != acc_batch_size. device:{device}:  epoch:{epoch}'+
                              f'total_size={total_size} -- i:{i} -- batch_no:{batch_no} of {len(train_loader)}')
                    continue

                step += 1
                grad_norm = torch.linalg.norm(get_grad_vector(model)) if step % 50 == 0 else np.nan
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # clip grads at 1.
                optimizer.step() # Updates the weights:  w = w - grad * lr
                optimizer.zero_grad(set_to_none=False)
                lr_scheduler.step()
                list_lr.append(optimizer.param_groups[-1]['lr'])
                torch.distributed.all_reduce(loss)
                loss /= world_size
                list_steps.append(step)
                list_losses.append(loss.item())
                list_secs.append(time.time() - s0)
                s0 = time.time()
                if is_main_process():
                    print(f' step:{step} -- loss:{list_losses[-1]:.2f}'+
                          f' -- lr:{list_lr[-1]:.4e}'+
                          f' -- grad_norm:{grad_norm:.2f}'+
                          f' -- num_tokens:{num_tokens:.2e}'+
                          f' -- {list_secs[-1]:.2f} secs'+
                          f' -- {print_runtime(start, False)}')

                if step % config['eval_iter'] == 0: # 10 steps
                    perplexity(model, device, val_data, list_steps_val, step, list_losses_val, list_ppl_val)

                if step % (config['eval_iter'] * 50) == 0:  # 500 steps
                    plotter(model, device, list_steps, list_losses, list_lr, list_ppl_val, list_steps_val, 
                            list_losses_val, list_secs, start)

                if step % (config['eval_iter'] * 200) == 0: # 2000 steps
                    generate_text(model, device, step, ppl=list_ppl_val[-1])
                    save_ckpt(device, model, optimizer, step)

