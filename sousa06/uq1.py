import os, sys, time
sys.path.append("/home/arbiter/qc")

from mod import (
                np, sp,
                mpl, plt,
                qc_dir,
                unpy,
                )
import scipy
from scipy import integrate
from scipy.constants import hbar, pi
from scipy.linalg import expm
from scipy.interpolate import spline
from itertools import takewhile, repeat
from multiprocess import Pool
from parallel import parallel
from multiprocess.dummy import Pool as ThreadPool
import linmax
import dill

hoa = 1.

tau_c_0 = 0.4 * hoa
tau_c_f = 31. * hoa
dtau_c = 0.54 * hoa
N = 700 # number of RTN trajectories
t_end = tau_c_f + 0.5 * hoa # end of RTN

tau_c = tau_c_0
tau_cs = [tau_c]
while tau_c < tau_c_f:
    tau_c += dtau_c
    tau_cs.append(tau_c)

ct = type(1.j)

fids_pi = np.loadtxt("data/fids_pi.txt", dtype=ct)
fids_C = np.loadtxt("data/fids_C.txt", dtype=ct)
fids_SC = np.loadtxt("data/fids_SC.txt", dtype=ct)

tx = []
print(len(fids_C))
tx.append(fids_C[0])
for i in range(1,len(fids_C) - 1):
    # if i % 2 == 0:
    #     continue
    avg = (fids_C[i-1] + fids_C[i] + fids_C[i+1]) / 3.
    mx = max([fids_C[i-1], fids_C[i], fids_C[i+1]])
    tx.append(avg)
    # tx.append(avg)
tx.append(fids_C[-1])
print(len(tx))
fids_C = np.array(tx)

tau_cs = np.array(tau_cs)
xnew = tau_cs
xnew = np.linspace(tau_cs.min(),tau_cs.max(),400)
fids_pi = spline(tau_cs, fids_pi, xnew)
fids_C = spline(tau_cs, fids_C, xnew)
fids_SC = spline(tau_cs, fids_SC, xnew)


plt.plot(xnew, fids_pi, 'b--', label="pi pulse")
plt.plot(xnew, fids_C, 'r-', label="CORPSE pulse")
plt.plot(xnew, fids_SC, 'r--', label="SCORPSE pulse")

plt.show()
