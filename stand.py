# from mod import (
#                 np, mpl, plt,
#                 os, sys,
#                 qc_dir,
#                 unpy,
#                 myAddition,
#
#                 )

from mod import *

print("Start: " + __file__)
print("unpy: " + unpy(__file__))

print(myAddition(3,5))

print(np.array([1,2]))

print(os.listdir(qc_dir)[0])

print(getNextName(unpy(absPath('qc1.py')),'py'))
