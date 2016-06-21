import os
import sys

import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from getch import getch

qc_dir = "/home/arbiter/qc/"
figures_rdir = "figures/"

def myAddition(x, y):
    return x + y + 2

def absPath(relPath):
    return qc_dir + relPath

def flipJoin(array, delimiter):
    """
    Joins array with delimiter.
    """

    return delimiter.join(array)

def unpy(pyfile):
    delim = '.'
    tokens = pyfile.split(delim)
    tokens.pop()
    return delim.join(tokens)

def addParens(s):
    return "(%s)" % (s)

# http://stackoverflow.com/questions/12375612/avoid-duplicate-file-names-in-a-folder-in-python
def getNextName(filepath, ext):
    uniq = 0
    filepath_mod = filepath + "." + ext
    while os.path.exists(filepath_mod):
        uniq += 1
        filepath_mod = "%s %s.%s" % (filepath, addParens(uniq), ext)
    return filepath_mod

def saveWith(relPath, savefunc, keyword_params):
    """
    Calls savefunc(reldir, [keyword_params])
    """
    if keyword_params == None:
        keyword_params = {}
    apply(savefunc, [absPath(relPath)], keyword_params)

def mapBothPend(before, mids, after):
    return [ "%s%s%s" % (before, s, after) for s in mids ]

def promptSaveFigs(names, figs):
    print("Enter 'y' to save figures. Discarded otherwise: ")
    saveFig = getch()
    if saveFig == "y":
        for i in range(len(names)):
            print("Saved as " + names[i])
            name = figures_rdir + names[i]
            fig = figs[i]
            saveWith(name, fig.savefig, None)
    else:
        print("Discarded.")


if __name__ == "__main__":
    print("mod.py run as __main__")
