import re
import os
import gc
import sys
import math
import time
import pickle
import random
import unicodedata
import collections
import numpy as np

from readers.Mention import Mention

def save(fname, obj):
    """
    Save a python object.

    Args:
        fname: (str): write your description
        obj: (todo): write your description
    """
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load(fname):
    """
    Load a pickle file.

    Args:
        fname: (str): write your description
    """
    with open(fname, 'rb') as f:
        return pickle.load(f)

def _getLnrm(arg):
    """Normalizes the given arg by stripping it of diacritics, lowercasing, and
    removing all non-alphanumeric characters.
    """
    arg = ''.join(
        [c for c in unicodedata.normalize('NFD', arg) if unicodedata.category(c) != 'Mn'])
    arg = arg.lower()
    arg = ''.join(
        [c for c in arg if c in set('abcdefghijklmnopqrstuvwxyz0123456789')])
    return arg

def load_crosswikis(crosswikis_pkl):
    """
    Loads the crosswik

    Args:
        crosswikis_pkl: (str): write your description
    """
    stime = time.time()
    print("[#] Loading normalized crosswikis dictionary ... ")
    crosswikis_dict = load(crosswikis_pkl)
    ttime = (time.time() - stime)/60.0
    print(" [#] Crosswikis dictionary loaded!. Time: {0:2.4f} mins. Size : {1}".format(
    ttime, len(crosswikis_dict)))
    return crosswikis_dict
#end

def load_widSet(
    widWikititle_file="/save/ngupta19/freebase/types_xiao/wid.WikiTitle"):
    """
    Loads widgets from a text file.

    Args:
        widWikititle_file: (str): write your description
    """
    print("Loading WIDs in the KB ... ")
    wids = set()
    with open(widWikititle_file, 'r') as f:
        text = f.read()
        lines = text.strip().split("\n")
        for line in lines:
            wids.add(line.split("\t")[0].strip())

    print("Loaded all WIDs : {}".format(len(wids)))
    return wids

def make_mentions_from_file(mens_file, verbose=False):
    """
    Make list of mentionions from a file.

    Args:
        mens_file: (str): write your description
        verbose: (bool): write your description
    """
    stime = time.time()
    with open(mens_file, 'r') as f:
        mention_lines = f.read().strip().split("\n")
        mentions = []
        for line in mention_lines:
            mentions.append(Mention(line))
            ttime = (time.time() - stime)
    if verbose:
        filename = mens_file.split("/")[-1]
        print(" ## Time in loading {} mens : {} secs".format(mens_file, ttime))
    return mentions


def get_mention_files(mentions_dir):
    """
    Return a list of the given directory.

    Args:
        mentions_dir: (str): write your description
    """
    mention_files = []
    for (dirpath, dirnames, filenames) in os.walk(mentions_dir):
        mention_files.extend(filenames)
        break
    #endfor
    return mention_files


def decrSortedDict(dict):
    """
    Decrides dict.

    Args:
        dict: (dict): write your description
    """
    # Returns a list of tuples (key, value) in decreasing order of the values
    return sorted(dict.items(), key=lambda kv: kv[1], reverse=True)

if __name__=="__main__":
    measureCrossWikisEntityConverage("/save/ngupta19/crosswikis/crosswikis.normalized.pkl")
