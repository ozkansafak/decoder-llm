from utils.scraper.imports import *


def extract_text_from_url(url, num_chars, data, visited_urls):
    if url in visited_urls:
        print(f'url already in visited_urls: {url}')
        return None, num_chars
    
    try:
        html = urlopen(url).read()
    except Exception as e:
        print(f'Exception:{e}, url: {url} ')
        visited_urls[url] = None
        return None, num_chars
    
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
    #print(f'num_characters text :{len(text): 9d} {url}')

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = unicodedata.normalize("NFKD", text)
    text = unidecode.unidecode(text)

    data[url] = text
    visited_urls[url] = None
    
    num_chars += len(data[url])
    if num_chars < 1e6:
        ptxt = f'{num_chars*1e-3:3.0f}K'
    elif num_chars < 1e9:
        ptxt = f'{num_chars*1e-6:3.0f}M'
    elif num_chars < 1e12:
        ptxt = f'{num_chars*1e-9:3.0f}G'

    print(f'chars added:{len(data[url])//1000:3d}K   num_chars:{ptxt}  {url}')

    return html, num_chars


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


def shave(new_links, visited_urls, cutoff=3):
    out = set()
    for item in new_links:
        # avoid loops in graph of urls
        if item not in visited_urls:
            out.add(item)
            if len(out) == cutoff:
                break
    return out


def write_data_to_pkl(data, fcount, visited_urls):
    # write data to pickle 
    fname = f'dataset/wiki_{fcount:03d}.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(data, file)
        print(f'data written to {fname}')

    for url in data.keys():
        visited_urls[url] = fname

    return dict(), fcount + 1


def decompose_divs(soup, list_class_names, name=''):
    if type(list_class_names) == str:
        list_class_names = [list_class_names]
    items = soup.find_all("div", {"class": list_class_names})
    for item in items:
        item.decompose()
        

