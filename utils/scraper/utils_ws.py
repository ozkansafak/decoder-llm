from utils.scraper.imports_ws import *
from utils.helpers import clean_up
from utils.helpers import vocab




def write_data_to_pkl(data, fcount, visited_urls):
    # write data to pickle 
    fname = f'dataset/wiki_{fcount:03d}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
        print(f'data written to {fname}')
        
    with open(f'dataset/visited_urls.txt', 'w') as f:
        json.dump(visited_urls, f)

    # initialize data
    data = dict()
    return data, fcount + 1


