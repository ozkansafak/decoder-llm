import os
import ipdb, re, pytz, datetime, time, sys, pickle, glob, json, random, unidecode, unicodedata
from collections import Counter
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
from pprint import pprint
import torch
from torch import nn
from torch.nn import functional as F
from prettytable import PrettyTable
import tiktoken 

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.imports import print_runtime, count_parameters, d_head, vocab_size, vocab, new_links, visited_urls, batch_size, d_model, n_heads, n_layer, block_size, learning_rate, dropout, max_steps, num_chars, add_gpu, encode, decode, plt, pylab, tokenizer


def load_val_data(device, world_size):
    """ len(val_data) = 500000
    """
    s0 = time.time()
    with open('dataset/val_data.json', 'r') as _f:
        val_data = json.load(_f)

    val_data = torch.tensor(val_data, dtype=torch.long, device=device)
    if device == 0:
        print(f'len(val_data):{len(val_data)}  {print_runtime(s0, False)}')
    
    return val_data


def extract_single_url(url, visited_urls, n_tokens):
    """ :param n_tokens: cumulative number of tokens crawled.
        :param visited_urls: visited_url[url] = len(text)
                             This function updates `visited_urls` in place.
    """

    if url in visited_urls:
        print(f'url already in visited_urls: {url}')
        return '', None, [], n_tokens

    try:
        html = urlopen(url).read()
    except Exception as e:
        print(f'Exception:{e}, url: {url}')
        visited_urls[url] = None
        return '', None, [], n_tokens

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
    tokens = encode(text)
    n_tokens += len(tokens)

    return text, html, tokens, n_tokens


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
    new_links = [url for url in links if url not in set_exclude]

    return new_links


def shave(device, new_links, visited_urls):
    
    remove = set([url for url in new_links if url in visited_urls])
    if remove:
        print(f'==> shave(): device:{device}, remove:{remove} !!!')

    for url in remove:
        new_links.remove(url)

    return


def decompose_divs(soup, list_class_names, name=''):
    if type(list_class_names) == str:
        list_class_names = [list_class_names]
    items = soup.find_all("div", {"class": list_class_names}) 
    for item in items:
        item.decompose()


def plotter(model, device, list_num_tokens, list_losses, list_lr, list_num_tokens_val, 
            list_losses_val, list_secs, start, savefig=True):

    if device != 0:
        return
    
    step = len(list_losses)
    list_num_tokens = np.array(list_num_tokens) / 1e6
    list_num_tokens_val = np.array(list_num_tokens_val) / 1e6
    
    fig = plt.figure(figsize=(3.5 * 1.618 * 2, 3.5 * 3))
    spec = fig.add_gridspec(3, 2, height_ratios=[1.5, 2, 2])
    ax00 = fig.add_subplot(spec[0, :])
    ax10 = fig.add_subplot(spec[1, 0])
    ax11 = fig.add_subplot(spec[1, 1])
    ax20 = fig.add_subplot(spec[2, 0])
    ax21 = fig.add_subplot(spec[2, 1])

    ax00.set_axis_off()
    ax00.text(0.13, .9, model.module.specs, ha='left', va='top', family='monospace', size='smaller')
    ax00.text(0.13, 0.5, model.module.table, ha='left', va='top', family='monospace', size='smaller')

    ax10.plot(list_num_tokens, list_losses, 'k', alpha=.6, label='train')
    ax10.plot(list_num_tokens_val, list_losses_val, 'r.-', alpha=.6, label='val')
    ax10.legend()
    ax10.set_title(f'Cross-Entropy Loss (step={step}) {print_runtime(start, False)} ')
    ax10.set_xlim(0)
    ax10.set_ylim(0)
    ax10.set_yticks(range(0, int(max(ax10.get_yticks()))))
    
    ax11.plot(list_num_tokens, list_lr, 'k.', alpha=.5, label='learning_rate')
    ax11.set_xlim(0)
    ax11.set_ylim(0)

    ax20.semilogy(list_num_tokens, list_secs[1], 'k', alpha=.2, label='time (one batch training)')
    ax20.semilogy(list_num_tokens, list_secs[1], 'k.', alpha=.5)
    ax20.semilogy(list_num_tokens_val, list_secs[0], 'r', alpha=.2, label='time (avg 30 steps)')
    ax20.semilogy(list_num_tokens_val, list_secs[0], 'r.', alpha=.5)
    ax20.set_xlim(0)
    ax20.set_xlabel('Million tokens')
    [ax.legend() for ax in [ax10, ax11, ax20]]
    
    if savefig:
        pst = pytz.timezone('US/Pacific')
        delta = - datetime.timedelta(hours=8) + datetime.timedelta(minutes=20) + datetime.timedelta(seconds=42)
        dt = datetime.datetime.now(pst) + delta
        prefix = dt.isoformat().split('.')[0]
        prefix = prefix.replace('T', ' | ')
        print(f'Saving "figures/loss_{prefix}.png"')
        plt.savefig(f'figures/loss_{prefix}.png', bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def clean_up(text, vocab):
    # replace out of vocab chars with ' '
    list_text = list(text)
    for i, c in enumerate(list_text):
        if c not in vocab:
            list_text[i] = ' '

    # idx: Remove the indices that are whitespaces and are followed by another white space.
    idx = [i for i in range(len(list_text)-1) if list_text[i] == list_text[i+1] == ' ']
    list_text = [list_text[i] for i in range(len(list_text)) if i not in set(idx)]

    text = ''.join(list_text)
    return text


def ptxt(num_chars):
    if num_chars < 1e6:
        txt = f'{num_chars*1e-3:3.2f} K'
    elif num_chars < 1e9:
        txt = f'{num_chars*1e-6:3.2f} M'
    elif num_chars < 1e12:
        txt = f'{num_chars*1e-9:3.2f} G'

    return txt


def crawl_urls(device, scraped_urls, visited_urls, n_tokens, add_gpu):
    """ :param add_gpu: number of characters to be crawled and added by each node.
    """

    s0 = time.time()
    head = 'https://www.wikipedia.org/wiki/'

    # initialize variables
    n_tokens_0 = n_tokens
    data = []
    shave(device, scraped_urls, visited_urls)

    page_no = 0
    if device == 0:
        print(f'BEGIN crawl_urls: device:{device},  add_gpu:{add_gpu/1e6:.3f} M,  n_tokens:{n_tokens/1e6:.2f} M')

    while n_tokens < n_tokens_0 + add_gpu:
        page_no += 1
        url = head + scraped_urls.pop(0)
        time.sleep(.3)
        text, html, tokens, n_tokens = extract_single_url(url, visited_urls, n_tokens)

        data.append(torch.tensor(tokens, dtype=torch.long))

    data = torch.cat(data)

    if device==0:
        print(f'END crawl_urls: device:{device}, add_gpu={add_gpu/1e6:.2f} M,  {len(data)*1e-6:.2f} M tokens, '+
              f'num pages: {page_no} '+
              f'{print_runtime(s0, False)}')

    data = data[:int(add_gpu)]  # crop the data at each GPU to same length, so training progresses in sync across all nodes.
    n_tokens = n_tokens_0 + add_gpu

    return data, n_tokens


























