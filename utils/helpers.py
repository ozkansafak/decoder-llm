from utils.imports import *
from utils.bigram import prepare_txt_data


def prepare_wiki_data(dirname='dataset/wikis/'):
    print(f'dirname: {dirname}')
    with open(fname, 'r') as f:
        text = f.read()

    # A quick look into the dataset
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
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


def plotter(list_epochs, list_losses, list_epochs_eval, list_losses_eval):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4*1.618, 4))
    ax.plot(list_epochs, list_losses, 'k', alpha=.6, label='train')
    ax.plot(list_epochs_eval, np.array(list_losses_eval)[:,0], 'b.-', alpha=.6, label='train')
    ax.plot(list_epochs_eval, np.array(list_losses_eval)[:,1], 'r.-', alpha=.6, label='val')
    ax.legend()
    ax.set_title('Cross-Entropy Loss')
    ax.set_xlabel('epochs')
    ax.set_xlim(0)
    ax.set_ylim(0)


def plot_character_frequency(urls, wikis, vocab):
    """ urls is a list of urls"""
    cnt = dict()
    for i, url in enumerate(urls):
        text = wikis[url]
        text = clean_up(text, vocab)
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
    
    
def clean_up(text, vocab):
    # replace out of vocab chars with ' '
    list_text = list(text)
    for i, c in enumerate(list_text):
        if c not in vocab:
            list_text[i] = ' '

    # idx: indices to be removed because they're followed by another white space.
    idx = [i for i in range(len(list_text)-1) if list_text[i] == list_text[i+1] == ' ']
    list_text = [list_text[i] for i in range(len(list_text)) if i not in set(idx)]
    
    text = ''.join(list_text)
    return text


