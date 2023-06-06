from utils.imports import *


vocab = set('\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
list_vocab = sorted(vocab)
vocab_size = len(vocab)
_stoi = {c:i for i, c in enumerate(list_vocab)}
_itos = {i:c for i, c in enumerate(list_vocab)}
_encode = lambda s: [_stoi[c] for c in s]  # takes in a string, output list of integers
decode = lambda inp: [_itos[i] for i in inp]  # input a list of integers, outputs a string


def extract_text_from_url(url, visited_urls, num_chars=0, printer=False):
    """ num_chars: cumulative number of characters in crawled.
    """
    
    if url in visited_urls:
        print(f'url already in visited_urls: {url}')
        return None, None, num_chars
    
    try:
        html = urlopen(url).read()
    except Exception as e:
        print(f'Exception:{e}, url: {url} ')
        visited_urls[url] = None
        return None, None, num_chars
    
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract() # rip it out

    # eliminate the references and infoboxes. 
    decompose_divs(soup, ["infobox"], name='infoboxes')
    decompose_divs(soup, ["mw-references-wrap", "mw-references-columns"], name='references')

    # get text
    divs = soup.find_all("div", class_='mw-parser-output')
    text = ''.join([div.get_text() for div in divs])

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # replace compatible characters with equivalents in English alphabet.
    text = unicodedata.normalize("NFKD", text)
    text = unidecode.unidecode(text)
    
    # clean up text. 
    text = clean_up(text, vocab)
    
    # write number of characters extracted to the visited_urls dictionary
    visited_urls[url] = len(text)
    num_chars += len(text)

    if num_chars < 1e6:
        ptxt = f'{num_chars*1e-3:3.0f}K'
    elif num_chars < 1e9:
        ptxt = f'{num_chars*1e-6:3.0f}M'
    elif num_chars < 1e12:
        ptxt = f'{num_chars*1e-9:3.0f}G'

    if printer: 
        print(f'chars added:{len(text)//1000:3d}K   num_chars:{ptxt}  {url}')
    else:
        print(f'chars added:{len(text)//1000:3d}K   num_chars:{ptxt}  {url}' + '  '*30, end='\r')
         
    return text, html, num_chars


def get_links(html):   
    if html is None:
        return None

    parser = BeautifulSoup(html, 'html.parser')
    links = [('https://www.wikipedia.org' + item.get('href'))
             for item in parser.find_all('a') 
             if item.get('href') and item.get('href').startswith('/wiki/')]

    # exclude URLs where there's a colon between two characters 
    # eg. https://www.wikipedia.org/wiki/Category:Articles_with_BNF_identifiers
    links = {url for url in links if not re.findall("[a-zA-Z0-9]:[a-zA-Z0-9]", url)}

    return links


def shave(new_links, visited_urls):
    remove = set()
    for url in new_links:
        # avoid loops in graph 
        if url in visited_urls:
            remove.add(url)
    print(f"shave(): removing urls that are already visited: ({len(remove)} urls)")
            
    for url in remove:
        new_links.remove(url)
    
    print(f'shave: len(new_links): {len(new_links)}')

    
def decompose_divs(soup, list_class_names, name=''):
    if type(list_class_names) == str:
        list_class_names = [list_class_names]
    items = soup.find_all("div", {"class": list_class_names})
    for item in items:
        item.decompose()


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


def prepare_streaming_wiki_data(text=None, printer=True):
    
    # single character tokenizer
    data = torch.tensor(_encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data


def crawl_wiki_data(new_links, num_chars, data, visited_urls, max_num_chars, printer=False):
    s0 = time.time()

    print(f'len(new_links): {len(new_links)},   len(visited_urls): {len(visited_urls)}' + '  '*50)
    shave(new_links, visited_urls)
    print('-----'*10  + '  '*50 + '\n')

    for i,url in enumerate(random.sample(new_links.copy(), k=len(new_links))):
        if url in visited_urls:
            new_links.remove(url)
        else:
            data[url], html, num_chars = extract_text_from_url(url, visited_urls, num_chars)
            new_links.remove(url)

            if html is not None:
                for item in get_links(html):
                    new_links.add(item)

        if i == 3:
            break

        # Terminate when max amount of text is crawled.
        if num_chars > max_num_chars:
            print(f'max characters reached: num_chars:{num_chars}. You should abort! {print_runtime(s0, printer=False)}')
            break

    print_runtime(s0, end='  '*50 +'\n')
    return num_chars


