import os, sys, time
sys.path.append("/home/arbiter/qc")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.constants import hbar, pi
from scipy.linalg import expm
from itertools import takewhile, repeat
from multiprocess import Pool
from multiprocess.dummy import Pool as ThreadPool
from math import cos, sin
import linmax
import dill

###
# Reproduction of:
# High-fidelity one-qubit operations under random telegraph noise
# Mikko Mottonen and Rogerio de Sousa
###

parallel = True
base = "data/report/"

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

hoa = (hbar / a_max)

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

###### Sym/Antisym pulses
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
    def X_sym(t):
        if t < 0:
            return 0
        elif t > tau:
            return 0
        _1 = theta / 2
        _2 = (a - _1) * cos((2 * pi  / tau) * t)
        _3 = a * cos((4 * pi  / tau) * t)
        return (_1 + _2 - _3)

    def X_antisym(t):
        if t < 0:
            return 0
        elif t > tau:
            return 0
        _1 = X_sym(t)
        _2 = b * sin((2 * pi  / tau) * t)
        _3 = (b/2) * sin((4 * pi  / tau) * t)
        return (_1 + _2 - _3)

    underlying = X_sym
    if antisym:
        underlying = X_antisym

    ma, mi = minmax(underlying, 0, tau)
    maxdiff = ma - mi
    def simplenorm(t):
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
def minmax(f, s, e):
    ts = np.linspace(s, e, 3000)
    fs = [(f(t)) for t in ts]
    ma = max(fs)
    mi = min(fs)
    return ma, mi

sym_pi = X_factory(pi, a1_asym, 0, False)
asym_pi = X_factory(pi, a1_asym, b1_asym, True)

# Systematic Error
def eta_sys(t):
    return Delta

# Eq. 2
# Hamiltonian
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
        return ((-1) ** sumHeavisideMonotonic(t, ts)) * eta_0
    return eta

# Can use takewhile over filter because monotonic
def sumHeavisideMonotonic(t, ts):
    return len([x for x in takewhile(lambda t_: t_ <= t,ts)])

# Eq. 8
def generateRho(rho_0, N, Us):
    def rho(t):
        def R(U_k):
            U = np.array(U_k(t))
            return np.dot(np.dot(U, rho_0), U.conj().T)
        if not parallel:
            terms = map(R, Us)
        else:
            terms = pool.map(R, Us)
        return (1./N) * sum(terms)
    return rho

# Eq. 9
# Generate unitary time evolution
identity = np.array([[1,0],[0,1]])
def generateU_k(a, eta_k, stepsize=0.03, t0=0., te=None, bch=5):
    def U_k(t):
        if t < t0:
            return identity
        if te is not None \
            and te < t:
            return identity

        steps = int(np.ceil(t / stepsize))
        ts = np.linspace(t0, t, steps)
        dt = ts[1] - ts[0]
        cvec = np.array(dt * -(1j), np.complex128)
        def G(t_):
            return np.dot(cvec, H_t(a(t_), eta_k(t_)))

        C = G(ts[0])
        for i in range(1, len(ts)):
            C = BCH(C, G(ts[i]), order=bch)

        return expm(C)
    return U_k

# Baker-Campbell-Hausdorff approx
# Order 5 is pretty necessary for accuracy
#   It takes about 85-90% more time.
#   Which translates to ~20-25% more time in total.
k1 = np.array(1/2.,np.complex128)
k2 = np.array(1/12.,np.complex128)
k3 = np.array(-1/12.,np.complex128)
k4 = np.array(-1/24.,np.complex128)
k5 = np.array(-1/720.,np.complex128)
k6 = np.array(1/360.,np.complex128)
k7 = np.array(1/120.,np.complex128)
def BCH(A, B, order=5):
    c1 = np.dot(k1,comm(A, B))
    c2 = np.dot(k2,comm(A, c1))
    c3 = np.dot(k3,comm(B, c1))
    res = A + B + c1 + c2 + c3
    if order > 3:
        c4 = np.dot(k4,comm(B, c2))
        res += c4
    if order > 4:
        c5 = np.dot(k5,(bch5(A, B) + bch5(B, A)))
        res += c5
    if order > 5:
        c6 = np.dot(k6,(bch6(A, B) + bch6(B, A)))
        res += c6
    if order > 6:
        c7 = np.dot(k7,(bch7(A, B) + bch7(B, A)))
        res += c7
    return res

def bch5(A, B):
    return comm(B,comm(B,comm(B,comm(B,A))))
def bch6(A, B):
    return comm(A,comm(B,comm(B,comm(B,A))))
def bch7(A, B):
    return comm(B,comm(A,comm(B,comm(A,B))))

# Commutator
def comm(A, B):
    return np.dot(A, B) - np.dot(B, A)

# Generate a rho
def ezGenerate_Rho(a, t_end, tau_c, eta_0, rho_0, N, stepsize=0.03):
    Us = [ezGenerateU_k(a, t_end, tau_c, eta_0, stepsize=stepsize) for i in range(N)]
    return (generateRho(rho_0, N, Us), Us)

# Generate a U_k
def ezGenerateU_k(a, t_end, tau_c, eta_0, stepsize=0.03):
    js = generateJumpTimes(t_end, tau_c)
    return generateU_k(a, generateEta(js, eta_0), stepsize=stepsize)

def ezGenerateEta(t_end, tau_c, eta_0):
    return generateEta(generateJumpTimes(t_end, tau_c), eta_0)

# Eq. 10
# Fidelity of a single transformation rho_0 -> rho_f
def fidSingleTxDirect(rho_f, rho, T):
    return np.trace(np.dot(rho_f.conj().T, rho(T)))

### GRAPE PULSE LOGIC

# Tricky with the fact that python lists start from 0
# Eq. 22
def makeLambda_mk(U_k, m, rho_f):
    n = len(U_k)
    # Product of U_k_n ... U_k_m+1
    U = U_k[n - 1]
    for j in range(n-2, m-1, -1):
        U = np.dot(U, U_k[j])
    return np.dot(np.dot(U.conj().T, rho_f), U)
# Eq. 23
def makeRho_mk(U_k, m, rho_0):
    # Product of U_k_m ... U_k_1
    U = U_k[m - 1]
    for j in range(m-2, -1, -1):
        U = np.dot(U, U_k[j])
    return np.dot(np.dot(U, rho_0), U.conj().T)

# Eq. 21
# T is end of the pulse, n is the number of "sections" (same width)
# 1 <= m <= n
def grad_phi_am(T, N, m, Us_mk, rho_0, rho_f):
    delta_t = T / n
    c = np.array((-1.j/2) * delta_t / N)

    terms = []
    res = 0
    def makeTerm(k):
        U_k = Us_mk[k]
        lambda_mk = makeLambda_mk(U_k, m, rho_f)
        lambda_H = lambda_mk.conj().T
        rho_mk = makeRho_mk(U_k, m, rho_0)
        term = np.trace(np.dot(lambda_H, comm(sigmaX, rho_mk)))
        return term
    terms = map(makeTerm, range(N))
    res = sum(terms)
    res = c * res
    return res

def GRAPE(T, n, N, rho_0, rho_f, tau_c, eta_0, stepsize, amps, epsilon=0.01):
    pulses = buildGrapePulses(amps, T)
    ts = np.linspace(0., T, n+1)
    Us_k = [] # Every U_k is actually a list of U_m
    new_amps = []

    def gen_Um(pulse, t0, te):
        js = generateJumpTimes((te - t0), tau_c)
        js = [j + t0 for j in js]
        return generateU_k(pulse, generateEta(js, eta_0, t0, te), stepsize=stepsize, t0=t0, te=te, bch=5)

    def genUs_m(m):
        t0 = ts[m]
        te = ts[m+1]
        pulse = pulses[m]
        t_avg = (t0 + te) / 2.
        U_m = gen_Um(pulse, t0, te)(te)
        return U_m

    for k in range(N):
        Us_m = map(genUs_m, range(n))
        Us_k.append(Us_m)

    grads = []

    def makeNewAmps(m):
        a_m = amps[m]
        d_phi_d_am = np.real(grad_phi_am(T, N, m, Us_k, rho_0, rho_f))
        grads.append(d_phi_d_am)
        # Eq. 13
        new_amp = a_m + epsilon*a_max*d_phi_d_am
        if new_amp <= -a_max:
            new_amp = .99 * -a_max
        if new_amp >= a_max:
            new_amp = .99 * a_max
        return new_amp

    new_amps = map(makeNewAmps, range(n))

    return new_amps


def pulseMaker(t0, te, amp):
    if t0 == 0.:
        def pulse(t):
            if t0 <= t and t <= te:
                return amp
            return 0.
        return pulse
    else:
        def pulse(t):
            if t0 < t and t <= te:
                return amp
            return 0.
        return pulse

def buildGrapePulses(amps, T):
    n = len(amps)
    ts = np.linspace(0, T, n + 1)
    pulses = []
    for m in range(1, len(ts)):
        t0 = ts[m-1]
        te = ts[m]
        amp = amps[m-1]
        pulse = pulseMaker(t0, te, amp)
        pulses.append(pulse)
    return pulses

def aggAmps(amps, T):
    def pulse(t):
        if t < 0 or T < t:
            return 0
        dt = T / float(len(amps))

        time_bin = int(np.floor(t / dt))
        if time_bin == len(amps):
            return amps[-1]
        return amps[time_bin]
    return pulse

def aggPulse(pulses):
    def pulse(t):
        return sum([pulse_section(t) for pulse_section in pulses])
    return pulse

####

def update_plots(fig, ax, plots, xss, yss):
    if len(plots) is not len(xss) \
        or len(plots) is not len(yss) \
        or len(xss) is not len(yss):
        raise ValueError("plots, xss, and yss need to be the same length!")

    for i in range(len(plots)):
        plot = plots[i]
        xs = xss[i]
        ys = yss[i]
        plot.set_xdata(xs)
        plot.set_ydata(ys)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    plt.pause(0.0001)
####

np.random.seed()

# Bit flip on computational basis
rho_0 = dm_1
rho_f = dm_0
eta_0 = Delta

tau_c_0 = 0.2 * hoa
tau_c_f = 32. * hoa
###
# Performance Params
###
dtau_c = 0.89 * hoa
N = 1300 # number of RTN trajN = ectories
stepsize = 0.023 # Step-forward matrices step size

###
t_end = tau_c_f + 0.42 * hoa # end of RTN

cpus = 8
if parallel:
    print("POOL")
    pool = Pool(processes=cpus)

tau_c = tau_c_0
tau_cs = [tau_c]
while tau_c < tau_c_f:
    tau_c += dtau_c
    tau_cs.append(tau_c)

T_G = 4 * hoa # sousa figure 2
n = 6 # number of different pulse amplitudes
epsilon = 0.6 # amount each gradient step can influence amps
grape_steps = 8 # number of optimization steps
### Grape
doGrape = True
if doGrape:
    tau_grape = 3.
    init_amps = [a_max for i in range(n)]
    grape_amps = init_amps
    for i in range(grape_steps):
        start = time.time()
        grape_amps = GRAPE(T_G, n, 50, rho_0, rho_f, tau_grape, eta_0, stepsize, grape_amps, epsilon)
        print("grapestep "+str(i)+": " + str(time.time() - start))
        print(grape_amps)
    np.savetxt(base + "grape_pulse.txt", grape_amps)
    # grape_amps = [-9.621202561946972098e-01, 8.504097303622385473e-01, 9.636089949467689930e-01, 9.616603542428837637e-01, 9.889276095541640332e-01, 9.688337027709256200e-01]
    grape_pulse = aggAmps(grape_amps, T_G)

eta_0_a_max = eta_0 / a_max
print("""
rho_0:
{rho_0}
rho_f:
{rho_f}
eta_0 / a_max:
    {eta_0_a_max}
number of RTN trajectories:
    {N}
Step-forward step size of:
    {stepsize}
tau_c going from {tau_c_0} to {tau_c_f}
tau step size of:
    {dtau_c}
end of RTN:
    {t_end}

Starting...
""".format(**locals()))

do_pi = True
do_c = True
do_sc = True
do_g = doGrape
do_sym = True
do_asym = False

fidelities = []
if do_pi:
    fids_pi = []
    fidelities.append(fids_pi)
if do_c:
    fids_C = []
    fidelities.append(fids_C)
if do_sc:
    fids_SC = []
    fidelities.append(fids_SC)
if do_g:
    fids_G = []
    fidelities.append(fids_G)
if do_sym:
    fids_sym = []
    fidelities.append(fids_sym)
if do_asym:
    fids_asym = []
    fidelities.append(fids_asym)

p_t = []
plt.ion()
fig, ax = plt.subplots()

sym_label = "Sym pulse, tau="+str(tau)
pulse_plots =[]
if do_pi:
    p_pi, = plt.plot(p_t, fids_pi, 'b--', label="pi pulse")
    pulse_plots.append(p_pi)
if do_c:
    p_c, = plt.plot(p_t, fids_C, 'r-', label="CORPSE pulse")
    pulse_plots.append(p_c)
if do_sc:
    p_sc, = plt.plot(p_t, fids_SC, 'r--', label="SCORPSE pulse")
    pulse_plots.append(p_sc)
if do_g:
    p_g, = plt.plot(p_t, fids_G, 'g-', label="GRAPE pulse")
    pulse_plots.append(p_g)
if do_sym:
    p_sym, = plt.plot(p_t, fids_sym, 'c--', label="Symmetric pulse")
    pulse_plots.append(p_sym)
if do_asym:
    p_asym, = plt.plot(p_t, fids_asym, 'c-', label=r"Symmetric pulse, $\tau$ = 9.325")
    pulse_plots.append(p_asym)

plt.xlabel(r"$\tau_c / (\hbar / a_{max})$")
plt.ylabel(r"$fidelity \phi(\rho_f, \rho_0)$")
plt.legend(loc='best')
plt.show()
plt.pause(0.0001)


start = time.time()
prev_time = -1
for i in range(len(tau_cs)):
    ministart = time.time()
    sys.stdout.write("\r"+str(i) + "/" + str(len(tau_cs)) + "  " + str(prev_time))
    print("\n")
    sys.stdout.flush()

    tau_c = tau_cs[i]
    p_t.append(tau_c)

    if do_pi:
        rho_pi, Us = ezGenerate_Rho(a_pi, t_end, tau_c, eta_0, rho_0, N, stepsize)
        fid_pi = fidSingleTxDirect(rho_f, rho_pi, T_pi)
        fids_pi.append(fid_pi)

    if do_c:
        rho_C, Us = ezGenerate_Rho(a_C, t_end, tau_c, eta_0, rho_0, N, stepsize)
        fid_C = fidSingleTxDirect(rho_f, rho_C, T_C)
        fids_C.append(fid_C)

    if do_sc:
        rho_SC, us = ezGenerate_Rho(a_SC, t_end, tau_c, eta_0, rho_0, N, stepsize)
        fid_SC = fidSingleTxDirect(rho_f, rho_SC, T_SC)
        fids_SC.append(fid_SC)

    if doGrape:
        rho_G, Us = ezGenerate_Rho(grape_pulse, t_end, tau_c, eta_0, rho_0, N, stepsize)
        fid_G = fidSingleTxDirect(rho_f, rho_G, T_G)
        fids_G.append(fid_G)

    if do_sym:
        rho_sym, us = ezGenerate_Rho(sym_pi, t_end, tau_c, eta_0, rho_0, N, stepsize)
        fid_sym = fidSingleTxDirect(rho_f, rho_sym, tau)
        fids_sym.append(fid_sym)

    if do_asym:
        rho_asym, us = ezGenerate_Rho(asym_pi, t_end, tau_c, eta_0, rho_0, N, stepsize)
        fid_asym = fidSingleTxDirect(rho_f, rho_asym, tau)
        fids_asym.append(fid_asym)

    update_plots(fig, ax, \
        pulse_plots, \
        [p_t for i in range(len(pulse_plots))], \
        fidelities )

    miniend = time.time()
    prev_time = miniend - ministart
print("time taken: " + str(time.time() - start))

if not os.path.exists(base):
    print "Creating directory (at end)"
    os.makedirs(base)

if do_pi:
    np.savetxt(base + "fids_pi.txt", fids_pi)
if do_c:
    np.savetxt(base + "fids_C.txt", fids_C)
if do_sc:
    np.savetxt(base + "fids_SC.txt", fids_SC)
if do_g:
    np.savetxt(base + "fids_G.txt", fids_G)
if do_sym:
    np.savetxt(base + "fids_sym.txt", fids_sym)
if do_asym:
    np.savetxt(base + "fids_asym.txt", fids_asym)

shapes.savefig(base + "shapes.png")
fig.savefig(base + "fig.png")

print("Done! Press Enter to exit.")
raw_input()

### End of main portion
