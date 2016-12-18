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
from math import cos, sin
import linmax
import dill

bch_time = [0.]
g_time = [0.]
a_time = [0.]
eta_time = [0.]
h_time = [0.]
th_time = [0.]
###
# Reproduction of:
# High-fidelity one-qubit operations under random telegraph noise
# Mikko Mottonen and Rogerio de Sousa
###

profiling = False
parallel = (not profiling) and False
base = "pulse_plots/"
if not os.path.exists(base):
    print "Creating directory"
    os.makedirs(base)


sigmaX = np.array([    [0., 1.]  ,
                        [1., 0.]  ], np.complex128)

sigmaY = np.array([    [0.,-1.j] ,
                        [1.j, 0.] ], np.complex128)

sigmaZ = np.array([    [1., 0.]  ,
                        [0.,-1.]  ], np.complex128)


####
hbar = 1.
# Maximum amplitude of control field
# set to hbar so that hbar/a_max = 1
a_max = hbar

# Noise strength
# 0.125 is the factor used in the paper
Delta = 0.125 * a_max

# Computational Bases
cb_0 = np.array([1, 0], np.complex128)
cb_1 = np.array([0, 1], np.complex128)

# Density Matrices
dm_0 = np.array([ cb_0, [0, 0] ], np.complex128).T
dm_1 = np.array([ [0, 0], cb_1 ], np.complex128).T

### Pulses

# hoa = (hbar / a_max)
hoa = 1.

# Eq. 15
# Pi pulse
T_pi = pi * hoa
def a_pi(t):
    if 0 <= t and t <= T_pi:
        return a_max
    return 0

# Eq. 16
# CORPSE
T_C = (13 * pi / 3) * hoa
def a_C(t):
    if t <= 0:
        return 0
    if t < ((pi / 3) * hoa):
        return a_max
    if t <= ((2 * pi) * hoa):
        return -a_max
    if t < (T_C):
        return a_max
    return 0

# Eq. 17
# SCORPSE, i.e. Short CORPSE
T_SC = (7 * pi / 3) * hoa
def a_SC(t):
    if t <= 0:
        return 0
    if t < ((pi / 3) * hoa):
        return -a_max
    if t <= ((2 * pi) * hoa):
        return a_max
    if t < (T_SC):
        return -a_max
    return 0
def a_SC2(t):
    if t <= 0:
        return 0
    if t < ((pi / 3) * hoa):
        return -a_max
    if t <= (5.18 * hoa):
        return a_max
    if t < (T_SC):
        return -a_max
    return 0

###### Sym/Antisym pulses
# 2.0 for "normal"
# ~4.9 for "capped", 3tauc < .99; 15tauc ~ .98; (useless)
# ~7.9 for normed, even worse than capped
# tau = 4.9 * hoa # Electron relaxation time; idk the "real" value :)
tau = 9.325 # max for full normalization
# See paper by S. Pasini for constants
a1_sym = -2.159224 * (1/tau)
a2_sym = -5.015588 * (1/tau)
a1_asym = 5.263022 * (1/tau)
b1_asym = 17.850535 * (1/tau)
a2_asym = -16.809353 * (1/tau)
b2_asym = 15.634390 * (1/tau)
# Makes X(t) driving pulse function
# theta is either pi or pi/2
# a, b are constants
# This pulse lasts a single period: 0 -> tau
def X_factory(theta, a, b, antisym):
    # norm = -2.0 * a + theta
    def X_sym(t):
        if t < 0:
            return 0
        elif t > tau:
            return 0
        _1 = theta / 2
        _2 = (a - _1) * cos((2 * pi  / tau) * t)
        _3 = a * cos((4 * pi  / tau) * t)
        # return minabs((_1 + _2 - _3) * a_max, a_max)
        return (_1 + _2 - _3)
        # return (_1 + _2 - _3) * a_max / norm # normed

    def X_antisym(t):
        if t < 0:
            return 0
        elif t > tau:
            return 0
        _1 = X_sym(t)
        _2 = b * sin((2 * pi  / tau) * t)
        _3 = (b/2) * sin((4 * pi  / tau) * t)
        # return minabs((_1 + _2 - _3) * a_max, a_max)
        return (_1 + _2 - _3)
        # return (_1 + _2 - _3) * a_max / norm # normed

    underlying = X_sym
    if antisym:
        underlying = X_antisym
    # norm = maximize(underlying, 0, tau)
    ma, mi = minmax(underlying, 0, tau)
    maxdiff = ma - mi
    def normmed(t):
        return underlying(t) / max(abs(ma),abs(mi))
    def fullnorm(t):
        if t < 0:
            return 0
        elif t > tau:
            return 0
        return (2*(underlying(t) - mi) / (maxdiff)) - a_max

    return fullnorm

def minabs(x, y):
    if abs(x) < abs(y):
        return x
    if x < 0:
        return -y
    return y

# maximize f from s to e
def maximize(f, s, e):
    ts = np.linspace(s, e, 3000)
    m = max([abs(f(t)) for t in ts])
    return m

# maximize f from s to e
def minmax(f, s, e):
    ts = np.linspace(s, e, 3000)
    fs = [(f(t)) for t in ts]
    ma = max(fs)
    mi = min(fs)
    return ma, mi

def x2p(width, periods):
    def pulse(t):
        if t < 0:
            return 0
        if t > width * periods:
            return 0
        # return (((t % width) - (width/2.0)) ** 2)/a_max
        return (((t % width) - (width/2.0)) ** 2)/(2*a_max) - a_max
    return pulse

# constant sample of a function from s to e
def csamplef(f, s, e, sections):
    bins = np.linspace(s, e, sections)
    dbin = bins[1] - bins[0]
    fsections = [f(t) for t in bins]
    def g(t):
        if t < 0:
            return 0
        if t >= e:
            return 0
        bin_num = int((t+dbin/2.) / dbin)
        return fsections[bin_num]
    return g


# Systematic Error
def eta_sys(t):
    return Delta

# Eq. 2
# Hamiltonian
def generateH(a, eta):
    def H_(t):
        return H_t(a(t), eta(t))
    return H_

def H_p(a, eta, t):
    if profiling:
        start = time.time()
        a_t = a(t)
        a_time[0] += time.time() - start
        return H_t2(a_t, eta(t))
    return H_t2(a(t), eta(t))

# Evaluate H given pre-evaluated a(t) and eta(t)
def H_t(a_t, eta_t):
    return H_a(a_t) + H_eta(eta_t)

def H_a(a_t):
    return 0.5 * a_t * sigmaX
def H_eta(eta_t):
    return 0.5 * eta_t * sigmaZ

# Eq. 4
# Jump times for RTN trajectory
# t_end can be understood in units of tau_c
# tau_c is the Noise correlation time
def generateJumpTimes(t_end, tau_c):
    ts = []
    t = 0.
    while t < t_end:
        p = np.random.random_sample()
        dt = (-tau_c) * np.log(p)
        t += dt
        ts.append(t)
    return ts

# Eq. 5
# Generate a noise function from jumps
# i.e. an RTN trajectory
# The heaviside function used is 1 at 0.
def generateEta(ts, eta_0, t0=None, te=None):
    def eta(t):
        if t0 is not None \
            and t < t0:
            return 0
        if te is not None \
            and te < t:
            return 0
        if t < ts[0]:
            return eta_0
        if profiling:
            start = time.time()
            res = ((-1) ** sumHeavisideMonotonic(t, ts)) * eta_0
            eta_time[0] += time.time() - start
            return res
        return ((-1) ** sumHeavisideMonotonic(t, ts)) * eta_0
    return eta

# Can use takewhile over filter because monotonic
def sumHeavisideMonotonic(t, ts):
    return len([x for x in takewhile(lambda t_: t_ <= t,ts)])

def ezGenerateEta(t_end, tau_c, eta_0):
    return generateEta(generateJumpTimes(t_end, tau_c), eta_0)


# XXX Shapes

def plotShape(pulsef, name, end=14*pi/3):
    shape = plt.figure()
    pulse_time = np.linspace(0, end, 5000)
    plt.plot(pulse_time, [pulsef(t) for t in pulse_time], "b-", label=name)
    plt.legend(loc="best")
    plt.ylabel(r"Amplitude in $a_{max}$")
    plt.xlabel(r"t in $\hbar/a_{max}$")
    plt.ylim([-1.1,1.1])
    plt.xlim([0.,end])
    filename = base + name.replace(" ", "_") + "-shape.png"
    shape.savefig(filename)

if not os.path.exists(base):
    print "Creating directory (at end)"
    os.makedirs(base)

# Pulses

# plotShape(a_pi, "Pi Pulse")
# plotShape(a_C, "CORPSE Pulse")
# plotShape(a_SC, "SCORPSE Pulse")

# Error Functions (Eta)

t_end = 20.

tau_c = 1.
eta1 = ezGenerateEta(t_end, tau_c, Delta)
plotShape(eta1, "Noise Function Tau_c 1 #1", t_end)
tau_c = 1.
eta1 = ezGenerateEta(t_end, tau_c, Delta)
plotShape(eta1, "Noise Function Tau_c 1 #2", t_end)
tau_c = 2.
eta1 = ezGenerateEta(t_end, tau_c, Delta)
plotShape(eta1, "Noise Function Tau_c 2", t_end)
tau_c = 4.
eta1 = ezGenerateEta(t_end, tau_c, Delta)
plotShape(eta1, "Noise Function Tau_c 4", t_end)

plt.show()

print("Pulses plotted under ./" + base)
