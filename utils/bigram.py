from utils.imports import *
from utils.helpers import * 
torch.manual_seed(1337)



assert int(n_embd // n_head) - (n_embd // n_head) == 0
print(f'device: {device}')
vocab = set('\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')


class SingleHead(nn.Module):
    """ one-head self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # idx and targets are both (B,T) tensor of integers
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Computer attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) / np.sqrt(C)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)  # randomly prevent nodes from communicating

        # Perform weighted aggregation of the values 
        v = self.value(x) # (B, T, T)
        out = wei @ v # (B, T, C)
        
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel 
        num_heads: each self-attention head is a communication channel for the tokens in sequence
        head_size: dimension of k,q,v vectors
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # as per the Attention paper, hidden layer size is 4x the input layer
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication (MultiHeadAttention) followed by computation (Feed Forward Network) """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__() 
        head_size = n_embd // n_head  # head size will be 8
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connections are a wonderful invention of modern science. Fork off and come back. 
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.compute_num_params()
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C) = (batch, time, vocab_size) = (4, 8, 65)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_embeddings + position_embeddings # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            if device.startswith('cuda'):
                loss = loss.mean()  

        return logits, loss

    def compute_num_params(self):
        num_params = 0
        for name, item in self.named_parameters():
            if len(item.shape) == 1:
                m = item.shape[0]
                num_params += m
            else:
                m, n = item.shape
            num_params += m * n

        if num_params < 1e3:
            print(f"num_params: {num_params}")
        elif num_params < 1e6:
            print(f"num_params: {int(num_params * 1e-3)}K")
        elif num_params < 1e9:
            print(f"num_params: {int(num_params * 1e-6)}M")
        elif num_params < 1e12:
            print(f"num_params: {int(num_params * 1e-9)}B")

        return num_params


def prepare_txt_data(fname='dataset/tiny_shakespeare.txt', text=None, printer=True):
    if fname:
        with open(fname, 'r') as f:
            text = f.read()  
    
    # A quick look into the dataset
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    if printer:
        print(fr"vocab:  {''.join(chars)}")
        print(f'vocab_size: {vocab_size}')

    # single character tokenizer
    stoi = {c:i for i, c in enumerate(chars)}
    itos = {i:c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # takes in a string, output list of integers
    decode = lambda inp: [itos[i] for i in inp]  # input a list of integers, outputs a string
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size, decode


def plot_character_frequency(urls, wikis):
    """ urls is a list of urls"""
    cnt = dict()
    for i, url in enumerate(urls):
        text = wikis[url]
        # text = clean_up(text, vocab)
        _, _, vocab_size, decode = prepare_txt_data(text=text, printer=False)
        cnt2 = Counter(text)
        
        for key,val in cnt2.items():
            if key not in cnt:
                cnt[key] = 0
            cnt[key] += cnt2[key]
        
        if i % 100 == 0:
            print(f'{i} of {len(urls)}', end='\r')

    cnt = sorted(cnt.items(), key=lambda x: x[1])
    y = []
    for a,b in cnt:
        y.append(b)

    plt.semilogy(y, 'k.')
    plt.xticks(range(len(y)), ''.join([f'{c}' for c,num in cnt]))
    plt.xlim(-0.2, len(y)-.8);

    return cnt
    
    
def generate(model, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]

        # get prediction 
        logits, _ = model(idx_cond)

        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)

        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)

        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    return idx


@torch.no_grad()  # tells torch we're never gonna call .backward() 
def estimate_loss(model, train_data, val_data, eval_iters, ib, start):
    losses = {}
    model.eval()
    for i, data in enumerate([train_data, val_data]):
        i_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, batch_size)
            logits, loss = model(xb, yb)

            if device.startswith('cuda') and torch.cuda.device_count() > 1:
                i_losses[k] = np.mean(loss.tolist())
            else:
                i_losses[k] = loss.item()
        losses['train' if i == 0  else 'val'] = i_losses.mean()

    print(f'step {str(ib)+":":5s} train_loss:{losses["train"]:.4f}, val_loss:{losses["val"]:.4f} {print_runtime(start, False)}')

    model.train()
    return losses


def get_batch(data, batch_size):
    """ gets batches at random
    """
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)  # move data to gpu if available

    return xb, yb


def get_batch_sequentially(data, batch_size, pivot):
    """   todo: get_batch_sequentially() needs to take as input one page and it should output number of batches mined.
                For now, I'm concatenating all the wiki pages in a single torch.tensor train_data.
                The last sentence of a page is preceded by the first sentence of the next page.
    """
    
    if len(data) - pivot < (batch_size * block_size):
        max_batches = len(data) // block_size
        
        print(f' ==> Not enough tokens in data. Need (batch_size * block_size) tokens in data, ' +
              f'but len(data) = {len(data)}, max_batches = {len(data) // block_size}')
        return None, None, 0
    
    ix = torch.arange(start=pivot, end=len(data) - block_size, step=block_size)[:batch_size]
    if len(ix) < batch_size:
        print('Beware!! Error coming up')
        ipdb.set_trace()
        
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)  # move data to gpu if available
    
    pivot += batch_size * block_size
    return xb, yb, pivot















































