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

hoa = 1.

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

# XXX PARAMETERS

do_pi = True
do_c = False
do_sc = True
do_sym = True
do_asym = False

tau_c_0 = 0.2 * hoa
tau_c_f = 30. * hoa
###
# Performance Params
###
dtau_c = 0.82 * hoa
N = 2700 # number of RTN trajectories
stepsize = 0.023 # Step-forward matrices step size, dont lower or raise

###
t_end = tau_c_f + 0.42 * hoa # end of RTN

profiling = False
cpus = 2
parallel = (not profiling) and True and False

###############


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

def donorm(underlying,s,e, normproc="simple"):

    ma, mi = minmax(underlying, s,e)
    maxdiff = abs(ma - mi)
    def simple(t):
        return underlying(t) / max(abs(ma), abs(mi))
    def full(t):
        return (2*underlying(t) / (maxdiff)) - a_max - (2*mi/maxdiff)
    def capped(t):
        return minabs(underlying(t), a_max)
    if normproc == "none":
        return underlying
    if normproc == "simple":
        return simple
    if normproc == "full":
        return full
    if normproc == "capped":
        return capped
    raise Exception("bad normproc: " + normproc)

###### Sym/Antisym pulses
# 2.0 for "normal"
# ~4.9 for "capped", 3tauc < .99; 15tauc ~ .98; (useless)
# ~7.9 for normed, even worse than capped
# tau = 4.9 * hoa # Electron relaxation time; idk the "real" value :)
# Makes X(t) driving pulse function
# theta is either pi or pi/2
# a, b are constants
# This pulse lasts a single period: 0 -> tau
def X_factory(theta, constPair, antisym, tau, normproc="simple",
                                                a=None, b=None):
    # See paper by S. Pasini for constants
    a1_sym = -2.159224 * (1/tau)
    a2_sym = -5.015588 * (1/tau)
    a1_asym = 5.263022 * (1/tau)
    b1_asym = 17.850535 * (1/tau)
    a2_asym = -16.809353 * (1/tau)
    b2_asym = 15.634390 * (1/tau)

    constfinder = { False: [None, (a1_sym,0), (a2_sym,0)],
                    True: [None, (a1_asym,b1_asym), (a2_asym,b2_asym)]}
    a_, b_ = constfinder[antisym][constPair]
    if a is None:
        a = a_
    if b is None:
        b = b_

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

    return donorm(underlying, 0, tau, normproc=normproc)

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

mytau = 9.12
sym_pi = X_factory(pi, 1, False, mytau,normproc="full")

def x2p(width, periods):
    def pulse(t):
        if t < 0:
            return 0
        if t > width * periods:
            return 0
        # return (((t % width) - (width/2.0)) ** 2)/a_max
        return (((t % width) - (width/2.0)) ** 2)/(2*a_max) - a_max
    return pulse

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

# time take = 269 without
# cpus = 8
# pool = Pool(processes=cpus)
# def multiMap(Us, t):
#     dec_Us = zip(Us, repeat(t))
#     return pool.map(evalMat, dec_Us)
# Eq. 8
def evalMat(dec_U_k):
    U_k, t = dec_U_k
    return np.array(U_k(t))
def generateRho(rho_0, N, Us):
    def rho(t):
        def R(U_k):
            U = np.array(U_k(t))
            return np.dot(np.dot(U, rho_0), U.conj().T)
        # U_k_ts = [np.array(U_k(t)) for U_k in Us]
        # terms = [U * rho_0 * U.conj().T for U in U_k_ts]
        if profiling or not parallel:
            terms = map(R, Us)
        else:
            terms = pool.map(R, Us)
        return (1./N) * sum(terms)
    return rho

# Eq. 9
# Generate unitary time evolution
# Ignore time-ordering for now...
def H_t2(a_t, eta_t):
    if profiling:
        start = time.time()
        res = 0.5 * a_t * sigmaX + 0.5 * eta_t * sigmaZ
        h_time[0] += time.time() - start
        return res
    return 0.5 * a_t * sigmaX + 0.5 * eta_t * sigmaZ
def w(a):
    return 3

U_count = [0.]
def generateU_k(a, eta_k, stepsize=0.03, t0=0., te=None):
    def U_k(t):
        if t < t0:
            return identity
        if te is not None \
            and te < t:
            return identity

        start = time.time()

        steps = int(np.ceil(t / stepsize))
        # steps = 700
        ts = np.linspace(t0, t, steps)
        dt = ts[1] - ts[0]
        cvec = np.array(dt * -(1j), np.complex128)
        def G(t_):
            if profiling:
                start = time.time()
                hp = H_p(a, eta_k, t_)
                teatime = time.time()
                th_time[0] += teatime - start
                res = np.dot(cvec, hp)
                g_time[0] += time.time() - teatime
                return res
            return np.dot(cvec, H_t2(a(t_), eta_k(t_)))

        C = G(ts[0])
        for i in range(1, len(ts)):
            # G_avg = 0.5 * (G(ts[i-1]) + G(ts[i]))
            C = BCH(C, G(ts[i]), order=5)

        # C = sum(Ss)
        # U_count[0] += 1

        end = time.time()
        delts = str(end - start)
        # sys.stdout.write("\rU++: " + str(U_count[0]) + "; " + delts)
        # sys.stdout.flush()
        # return linmax.powerexp(C)
        return expm(C)
    return U_k

k1 = np.array(1/2.,np.complex128)
k2 = np.array(1/12.,np.complex128)
k3 = np.array(-1/12.,np.complex128)
k4 = np.array(-1/24.,np.complex128)
k5 = np.array(-1/720.,np.complex128)
k6 = np.array(1/360.,np.complex128)
k7 = np.array(1/120.,np.complex128)
# Baker-Campbell-Hausdorff approx
# Order 5 is pretty necessary for accuracy
#   It takes about 85-90% more time.
#   Which translates to ~20-25% more time in total.
# Order 6 takes a long time.
def BCH(A, B, order=4):
    start = time.time()

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
    if profiling:
        bch_time[0] += time.time() - start
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
    # Us = repeat(ezGenerateU_k(a, t_end, tau_c, eta_0, stepsize=stepsize),N)
    return (generateRho(rho_0, N, Us), Us)

# Generate a U_k
def ezGenerateU_k(a, t_end, tau_c, eta_0, stepsize=0.03):
    js = generateJumpTimes(t_end, tau_c)
    return generateU_k(a, generateEta(js, eta_0), stepsize=stepsize)
    # return generateU_k(a, eta_sys)

def ezGenerateEta(t_end, tau_c, eta_0):
    return generateEta(generateJumpTimes(t_end, tau_c), eta_0)

# Eq. 10
# Fidelity of a single transformation rho_0 -> rho_f
def fidSingleTxDirect(rho_f, rho, T):
    return np.trace(np.dot(rho_f.conj().T, rho(T)))

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

if not profiling and parallel:
    print("POOL")
    pool = Pool(processes=cpus)

tau_c = tau_c_0
tau_cs = [tau_c]
while tau_c < tau_c_f:
    tau_c += dtau_c
    tau_cs.append(tau_c)

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
if do_sym:
    fids_sym = []
    fidelities.append(fids_sym)
if do_asym:
    fids_asym = []
    fidelities.append(fids_asym)


# Pulse Info: (amp_function, time_done)
# TODO: clean up different cases :)
# def mkInfo(ampf, time):
#     return {
#         ampf: ampf,
#         time: time,
#         fids: []
#     }
# pulse_infos = {
#     pi: mkInfo(a_pi, T_pi),
#     corpse: mkInfo(a_C, T_C),
#     scorpse: mkInfo(a_SC, T_SC),
#     sym: mkInfo(sym_pi, tau)
#     }

p_t = []

plt.figure()
chi_time = np.linspace(0, mytau, 500)
if do_sym:
    plt.plot(chi_time, [sym_pi(t) for t in chi_time], "c-", label="Symmetric Pulse shape")
if do_asym:
    plt.plot(chi_time, [asym_pi(t) for t in chi_time], "m-", label="Antisymmetric Pulse shape")
plt.legend(loc="best")

plt.ion()
fig, ax = plt.subplots()

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
    #r"$f(x) = x^2$; width $\pi$, 2 periods, range $-a_{max}$ to $a_{max}$"
if do_sym:
    p_sym, = plt.plot(p_t, fids_sym, 'c--', label="Symmetric pulse")
    pulse_plots.append(p_sym)
if do_asym:
    p_asym, = plt.plot(p_t, fids_asym, 'm--', label="Antisymmetric pulse")
    pulse_plots.append(p_asym)

plt.xlabel(r"$\tau_c / (\hbar / a_{max})$")
plt.ylabel(r"$\phi(\rho_f, \rho_0)$")
plt.legend(loc='best')
plt.show()
plt.pause(0.0001)

### XXX loop

start = time.time()
prev_time = -1
for i in range(len(tau_cs)):
    ministart = time.time()
    # if i % 15 is 0:
    sys.stdout.write("\r"+str(i) + "/" + str(len(tau_cs)) + "  " + str(prev_time))
    print("\n")
    sys.stdout.flush()

    if profiling:
        print("""
        g time: {g_time[0]}
        a time: {a_time[0]}
        eta time: {eta_time[0]}
        h time: {h_time[0]}
        th time: {th_time[0]}
        BCH time: {bch_time[0]}
        """.format(**locals()))
        print("\n")
        U_count[0] = 0
        a_time[0] = 0.
        eta_time[0] = 0.
        g_time[0] = 0.
        bch_time[0] = 0.
        h_time[0] = 0.
        th_time[0] = 0.

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

if do_pi:
    np.savetxt("data/q3/fids_pi.txt", fids_pi)
if do_c:
    np.savetxt("data/q3/fids_C.txt", fids_C)
if do_sc:
    np.savetxt("dataq/q3/fids_SC.txt", fids_SC)
if do_sym:
    np.savetxt("data/q3/fids_sym.txt", fids_sym)
if do_asym:
    np.savetxt("data/q3/fids_asym.txt", fids_asym)

fig.savefig("data/q3/fig.png")

print("Done! Press Enter to exit.")
raw_input()
