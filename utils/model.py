from utils.imports import *
from utils.helpers import * 
torch.manual_seed(1337)

print(f'device: {device}')
vocab = set('\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')


class SingleHead(nn.Module):
    """ one-head self-attention"""
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # idx and targets are both (B,T) tensor of integers
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Compute attention scores ("affinities")
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
        d_head: dimension of k,q,v vectors
    """
    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead() for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            # as per the Attention paper, hidden layer size is 4x the input layer
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication (MultiHeadAttention) followed by computation (Feed Forward Network) """

    def __init__(self, d_model, n_heads):
        # d_model: embedding dimension, n_heads: the number of heads we'd like
        super().__init__() 
        self.sa = MultiHeadAttention(n_heads)
        self.ffwd = MLP(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # residual connections are a wonderful invention of modern science. Fork off and come back. 
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


class DecoderModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model) # final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.print_hyperparams_to_stdout()
        
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

    def print_hyperparams_to_stdout(self):
        num_params = 0
        for name, item in self.named_parameters():
            if len(item.shape) == 1:
                m = item.shape[0]
                num_params += m
            else:
                m, n = item.shape
                num_params += m * n

        print()
        if num_params < 1e3:
            print(f"num_params: {num_params}")
        elif num_params < 1e6:
            print(f"num_params: {int(num_params * 1e-3)}K")
        elif num_params < 1e9:
            print(f"num_params: {int(num_params * 1e-6)}M")
        elif num_params < 1e12:
            print(f"num_params: {int(num_params * 1e-9)}B")

        print(f'd_model: {d_model}')
        print(f'n_layer: {n_layer}')
        print(f'n_heads:  {n_heads}')
        print(f'd_head:  {d_head}')
        print(f'block_size:  {block_size}')
        print(f'batch_size: {batch_size}')
        print(f'learning_rate:  {learning_rate}')
        print()
        
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


def generate_text(model, step):
    
    s0 = time.time()
    
    print(f'===>  Text Generation: ')
    
    print(f''.join(decode(generate(model, torch.ones((1,1), device=device, dtype=torch.long) * 35, 
                                   max_new_tokens=400)[0].tolist()))) 
    print_runtime(s0)
    print('---' *30)


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
def estimate_loss(model, train_data, val_data, step, start):
    s0 = time.time()
    losses = {}
    model.eval()
    xb = None
    
    print('\nestimating loss... ', end='\r')
    for i, data in enumerate([train_data, val_data]):
        pivot = 0
        i_losses = [] 
        while pivot == 0 or len(xb) == batch_size:
            xb, yb, pivot = get_batch(data, batch_size, pivot)
            logits, loss = model(xb, yb)
            i_losses.append(np.mean(loss.tolist()))
        losses['train' if i == 0  else 'val'] = np.mean(i_losses)
        
    print(f'train_loss:{losses["train"]:.4f}, val_loss:{losses["val"]:.4f} {print_runtime(s0, False)}')

    model.train()
    return losses


def get_batch_at_random(data, batch_size):
    """ gets batches at random.
    """

    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)  # move data to gpu if available

    return xb, yb


def get_batch(data, batch_size, pivot=0):
    """   todo: Take one page as input.
    """

    if len(data) - pivot < (batch_size * block_size):
        procured_batches = (len(data) - pivot) // block_size
        print(f' ==> get_batch: procured_batches = {procured_batches} out of {batch_size} requested')
        if procured_batches == 0:
            xb = torch.zeros((0, block_size), dtype=torch.long)
            yb = torch.zeros((0, block_size), dtype=torch.long)
            xb, yb = xb.to(device), yb.to(device)  # move data to gpu if available
            pivot = 0
            return xb, yb, pivot
    
    ix = torch.arange(start=pivot, end=len(data) - block_size, step=block_size)[:batch_size]
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)  # move data to gpu if available

    pivot += batch_size * block_size      
    return xb, yb, pivot


def load_train_objs(vocab_size, device, learning_rate):

    # instantiate model
    model = DecoderModel(vocab_size)
    model = model.to(device) # move model parameters to gpu if available

    # Create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # usually 3e-4 for bigger networks.

    return model, optimizer


def load_val_data(num_pages):

    # load val_data by crawling the list of wiki pages in "dataset/val_wiki.json"
    val_data, val_urls = load_val_data(num_pages)
    print(f'val_data:{val_data}')
    return val_data, val_urls


def train(model, optimizer, device, batch_size, block_size, xb, yb, num_chars, val_data,
          list_epochs, list_losses, list_epochs_eval, list_losses_eval, eval_num_samples):
    """ train loop 
    """

    start = time.time()
    step = 0
    train_data, num_chars = crawl_wiki_data(new_links, visited_urls, num_chars, add, printer=False)
    xb, yb, pivot = get_batch(train_data, batch_size, pivot=0)

    while step < max_iters:
        step += 1
        num_tokens = step * batch_size * block_size  #  num of tokens ingested.
        sample_no = step * batch_size * torch.cuda.device_count()

        while len(xb) < batch_size:
            repo_xb, repo_yb, pivot = xb, yb, 0
            train_data, num_chars = crawl_wiki_data(new_links, visited_urls, num_chars, add, printer=False)

            # Sample a batch of data to complete to the "batch_size"
            xb, yb, pivot = get_batch(train_data, batch_size - len(xb), pivot)
            xb = torch.cat((repo_xb, xb))
            yb = torch.cat((repo_yb, yb))

        # Evaluate the loss
        logits, loss = model(xb, yb)

        # Average if trained on multi-GPUs 
        loss = loss.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward() # get the gradients with backprop.
        optimizer.step() # apply the gradient on the parameters

        list_losses.append(loss.item())
        list_num_tokens.append(sample_no)

        # evaluate at fixed intervals
        if sample_no % (eval_num_samples//4) == 0:
            list_num_tokens_eval.append(sample_no)
            out = estimate_loss(model, train_data[-len(val_data):], val_data, step, start)
            list_losses_eval['val'].append(out['val'])
            list_losses_eval['train'].append(out['train'])
            plotter(list_num_tokens, list_losses, list_num_tokens_eval, list_losses_eval)

        if sample_no % (eval_num_samples * 4) == 0:
            generate_text(model, step, start)

        if step % 10 == 0: 
            print(f'step:{step:3d}  loss:{loss.item():.3f}  {print_runtime(start, False)}', end='\r')





































