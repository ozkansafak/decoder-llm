import re, sys, random, glob, json, unidecode, unicodedata
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import ipdb, time, sys, os, pickle
from urllib.request import urlopen
random.seed(42)
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from collections import Counter


pylab.rcParams.update({'legend.fontsize': 'large',
                       'font.size'      : 14,
                       'figure.figsize' : (18, 4),
                       'axes.labelsize' : 'large',
                       'axes.titlesize' : 'large',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'large',
                       'ytick.labelsize': 'large'})


def print_runtime(start, printer=True):
    end = time.time()
    if printer:
        print(f'Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec')
        return None
    else:
        return f' (...Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec)'


