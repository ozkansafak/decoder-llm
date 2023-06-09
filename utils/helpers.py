from utils.imports import *


# request children pages wiki pages starting with the root page.
max_num_chars = 100 * 1e6
num_chars = 0
visited_urls = dict()
url = "https://www.wikipedia.org/wiki/David_Bowie"
new_links = [url]
# ------------------------------------------------

vocab = set('\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
list_vocab = sorted(vocab)
vocab_size = len(vocab)
_stoi = {c:i for i, c in enumerate(list_vocab)}
_itos = {i:c for i, c in enumerate(list_vocab)}
_encode = lambda s: [_stoi[c] for c in s]  # takes in a string, output list of integers
decode = lambda inp: [_itos[i] for i in inp]  # input a list of integers, outputs a string


def load_val_data(num_pages=20, printer=False):
    with open('dataset/val_wiki.json', 'r') as _f:
        val_urls = json.load(_f)
        
    num_chars = 0
    val_data = []
    for i, url in enumerate(val_urls[:num_pages]):
        text, html, num_chars = extract_single_url(url, visited_urls, num_chars)
        val_data.append(torch.tensor(_encode(text), dtype=torch.long))
        if printer: 
            print(f'{i:2d}  {url}')
    
    val_data = torch.cat(val_data)

    
    return val_data, val_urls[:num_pages]


def extract_single_url(url, visited_urls, num_chars):
    """ :param num_chars: cumulative number of characters in crawled.
        :param visited_urls: visited_url[url] = len(text)
                             This function updates `visited_urls` in place.
    """

    if url in visited_urls:
        print(f'url already in visited_urls: {url}')
        return '', None, num_chars

    try:
        html = urlopen(url).read()
    except Exception as e:
        print(f'Exception:{e}, url: {url} ' + '  '*30)
        visited_urls[url] = None
        return '', None, num_chars

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

    return text, html, num_chars


def get_links(html, new_links, visited_urls):  
    if html is None:
        return []

    parser = BeautifulSoup(html, 'html.parser')
    links = {('https://www.wikipedia.org' + item.get('href'))
             for item in parser.find_all('a') 
             if item.get('href') and item.get('href').startswith('/wiki/')}

    # exclude URLs where there's a colon between two characters 
    # eg. https://www.wikipedia.org/wiki/Category:Articles_with_BNF_identifiers
    links = [url for url in links if not re.findall("[a-zA-Z0-9]:[a-zA-Z0-9]", url)]
    
    # only retain URLs that are not in `visited_url`
    set_exclude = set(visited_urls.keys()).union(set(new_links))
    links = [url for url in links if url not in set_exclude]

    return links


def shave(new_links, visited_urls):
    
    remove = set([url for url in new_links if url in visited_urls])
        
    for url in remove:
        new_links.remove(url)

    #print(f"shave(): No of visited urls removed: {len(remove)} urls" + '  ' * 50)

    return

    
def decompose_divs(soup, list_class_names, name=''):
    if type(list_class_names) == str:
        list_class_names = [list_class_names]
    items = soup.find_all("div", {"class": list_class_names})
    for item in items:
        item.decompose()


def plotter(list_epochs, list_losses, list_epochs_eval, list_losses_eval, step=None):
    list_epochs = np.array(list_epochs) / 1e6
    list_epochs_eval = np.array(list_epochs_eval) / 1e6
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3*1.618, 3))
    ax.plot(list_epochs, list_losses, 'k', alpha=.6, label='train')
    ax.plot(list_epochs_eval, np.array(list_losses_eval)[:,0], 'b.-', alpha=.6, label='train')
    ax.plot(list_epochs_eval, np.array(list_losses_eval)[:,1], 'r.-', alpha=.6, label='val')
    ax.legend()
    ax.set_title(f'Cross-Entropy Loss (step={step})')
    ax.set_xlabel('milion tokens')
    ax.set_xlim(0)
    ax.set_ylim(0)
    plt.show()


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


def ptxt(num_chars):

    if num_chars < 1e6:
        txt = f'{num_chars*1e-3:3.2f}K'
    elif num_chars < 1e9:
        txt = f'{num_chars*1e-6:3.2f}M'
    elif num_chars < 1e12:
        txt = f'{num_chars*1e-9:3.2f}G'

    return txt


def crawl_wiki_data(new_links, visited_urls, num_chars, add=5e5, step=None, printer=False):
    """ 
    """
    s0 = time.time()
    print(f'crawl_wiki_data at step={step}, add={add/1e6:.2f}M characters')

    # initialize variables
    num_chars_init = num_chars
    train_data = []
    n_init = len(new_links)

    if printer: print(f'num_chars_init:{num_chars_init}  ' +
                      f'len(new_links):{len(new_links)}, len(visited_urls):{len(visited_urls)}' + '  '*50 + '\n')

    shave(new_links, visited_urls)

    while num_chars < num_chars_init + add:

        url = new_links.pop(0)
        if url in visited_urls:
            print(f'WARNING: url in visited_urls!!!    ')
            stop_execution
            
        # shuffle new_links in place
        random.shuffle(new_links)

        text, html, num_chars = extract_single_url(url, visited_urls, num_chars)
        new_links.extend(get_links(html, new_links, visited_urls))
        train_data.append(torch.tensor(_encode(text), dtype=torch.long))

        if printer: print(f'page_length:{len(text)/1000:5.1f}K, '+
                          f'len(new_links):{len(new_links)}, len(visited_urls):{len(visited_urls)}, '+ 
                          f'num_chars:{ptxt(num_chars)}  {url}' + 
                          ' '*70
                          , end='\n')

    """   todo: get_batch_sequentially() needs to take as input one page and it should output number of batches mined.
                For now, I'm concatenating all the wiki pages in a single torch.tensor train_data.
                The last sentence of a page is preceded by the first sentence of the next page.
    """
    
    train_data = torch.cat(train_data).to(device)

    if printer: print(f'Exiting crawl_wiki_data(): '+
                      f'len(new_links): {len(new_links)}  '+
                      f'len(visited_urls):{len(visited_urls)}  '+
                      f'{print_runtime(s0, False)}' + '  '*50)

    return train_data, num_chars

















