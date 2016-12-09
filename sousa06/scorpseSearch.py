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

###### Sym/Antisym pulses
# 2.0 for "normal"
# ~4.6 for "capped"
# ~7.8 for normed
tau = (5.1 * pi / 3) * hoa # Electron relaxation time; idk the "real" value :)
# See paper by S. Pasini for constants
a1_sym = -2.159224
a2_sym = -5.015588
a1_asym = 5.263022
b1_asym = 17.850535
a2_asym = -16.809353
b2_asym = 15.634390 # * (1/tau)
# Makes X(t) driving pulse function
# theta is either pi or pi/2
# a, b are constants
# This pulse lasts a 1.5 periods: 0 -> tau * 1.5
# Why 1.5? You will notice the error-correcting pulses, SCORPSE
# and CORPSE, do so.
def X_factory(theta, a, b, antisym, tau=tau):
    norm = -2.0 * a + theta
    def X_sym(t):
        if t < 0:
            return 0
        elif t > tau*1.5:
            return 0
        t += tau * 0.75 # start at a min point, like SCORPSE
        _1 = theta / 2
        _2 = (a - _1) * cos((2 * pi  / tau) * t)
        _3 = a * cos((4 * pi  / tau) * t)
        # return minabs((_1 + _2 - _3) * a_max, a_max) # capped
        return (_1 + _2 - _3) * a_max / norm # normed
        # return invertcap((_1 + _2 - _3) * a_max, a_max) # inverted

    def X_antisym(t):
        if t < 0:
            return 0
        elif t > tau:
            return 0
        _1 = X_sym(t)
        _2 = b * sin((2 * pi  / tau) * t)
        _3 = (b/2) * sin((4 * pi  / tau) * t)
        # return minabs((_1 + _2 - _3) * a_max, a_max)
        return (_1 + _2 - _3) * a_max / norm
        # return invertcap((_1 + _2 - _3) * a_max, a_max) # inverted

    # def s(t):
    #     if t <= 0 or t > pi:
    #         return 0
    #     return a_max
    # return s
    if antisym:
        return X_antisym
    return X_sym

def minabs(x, y):
    if abs(x) < abs(y):
        return x
    if x < 0:
        return -y
    return y

def invertcap(x, y):
    return pow(-1, int(x/y))* x % y

# sym_pi = X_factory(pi, a1_asym, b1_asym, True)

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
# js = generateJumpTimes(15, 1)
# e = generateEta(js, 1)
# print([e(t/10.) for t in range(0, 110)])
# et = integrate.quad(e, 0, 10)
# print(et)


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

# Fidelity of a single transformation rho_0 -> rho_f
def fidSingleTx(rho_0, N, Us, T, rho_f):
    rho = generateRho(rho_0, N, Us)
    return fidSingleTxDirect(rho_f, rho, T)

# Eq. 10
# Fidelity of a single transformation rho_0 -> rho_f
def fidSingleTxDirect(rho_f, rho, T):
    return np.trace(np.dot(rho_f.conj().T, rho(T)))

# Eq. 11
def fidSingleTxFull(rho_0, rho_f, T, N, Us):
    U_k_ts = [np.array(U_k(T), np.complex128) for U_k in Us]
    mats = [np.dot(np.dot(rho_f.conj().T, U_k_t), np.dot(rho_0, U_k_t.conj().T)) for U_k_t in U_k_ts]
    terms = [np.trace(mat) for mat in mats]
    return (1./N) * sum(terms)

# Eq. 14
def fid14(Uf, N, Uks):
    c = np.array(1./(12*N))
    sigs = [sigmaX, sigmaY, sigmaZ]
    UfH = Uf.conj().T
    def term_func(Uk):
        UkH = Uk.conj().T
        res = 0.
        for sig in sigs:
            red = reduce(np.dot, [Uf, sig, UfH, Uk, sig, UkH])
            res += np.trace(red)
        return res
    terms = map(term_func, Uks)
    res = np.array(1/2.) + c * sum(terms)
    return res

identity = np.array([[1,0],[0,1]])


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
tau_c_f = 15. * hoa
times = [0.2* hoa, 3.0* hoa, 15.0* hoa]
###
# Performance Params
###
N = 420 # number of RTN trajectories
stepsize = 0.024 # Step-forward matrices step size, dont lower

###
t_end = 15.42 # end of RTN

cpus = 8
if not profiling and parallel:
    print("POOL")
    pool = Pool(processes=cpus)
#
# tau_c = tau_c_0
# tau_cs = [tau_c]
# while tau_c < tau_c_f:
#     tau_c += dtau_c
#     tau_cs.append(tau_c)
# tau_cs =
tau_start = ( 2.9 * pi / 3 )* hoa
tau_end = ( 14. * pi / 3 )* hoa
dtau = 0.12 * hoa
t_ = tau_start
taus = []
while t_ < tau_end:
    taus.append(t_)
    t_ += dtau
# sym_pis = [X_factory(pi, a1_sym, 0, False, tau=t_) for t_ in taus]
sym1 = []
sym3 = []
sym15 = []

eta_0_a_max = eta_0 / a_max
print("""
Starting...
""".format(**locals()))

def ezmap(f, xs):
    if parallel:
        return pool.map(f, xs)
    return map(f, xs)

p_t = []

plt.ion()
fig, ax = plt.subplots()

p1, = plt.plot(p_t, sym1, 'b--', label="t=1")
p3, = plt.plot(p_t, sym3, 'r-', label="t=3")
p15, = plt.plot(p_t, sym15, 'g--', label="t=15")

plt.xlabel("tau in (hbar / a_max)")
plt.ylabel("fidelity \\phi(rho_f, rho_0)")
plt.legend(loc='best')
plt.show()
plt.pause(0.0001)


start = time.time()
prev_time = -1
for i in range(len(taus)):
    tau = taus[i]

    p_t.append(tau)
    sym_pi = X_factory(pi, a2_sym / tau, 0, False, tau=tau)
    sym_pi = X_factory(pi, a1_asym / tau, b1_asym/tau, True, tau=tau)
    # sym_pi = X_factory(pi, a2_asym / tau, b2_asym/tau, True, tau=tau)
    sym_pi = X_factory(pi, a1_sym / tau, 0, False, tau=tau)

    rho_sym, us = ezGenerate_Rho(sym_pi, t_end, times[0], eta_0, rho_0, N, stepsize)
    fid_sym = fidSingleTxDirect(rho_f, rho_sym, tau)
    sym1.append(fid_sym)

    if fid_sym > 0.990:
        print("1@ " + str(tau) + ", " + str(fid_sym))

    rho_sym, us = ezGenerate_Rho(sym_pi, t_end, times[1], eta_0, rho_0, N, stepsize)
    fid_sym = fidSingleTxDirect(rho_f, rho_sym, tau)
    sym3.append(fid_sym)
    if fid_sym > 0.990:
        print("3@ " + str(tau) + ", " + str(fid_sym))

    rho_sym, us = ezGenerate_Rho(sym_pi, t_end, times[2], eta_0, rho_0, N, stepsize)
    fid_sym = fidSingleTxDirect(rho_f, rho_sym, tau)
    sym15.append(fid_sym)
    if fid_sym > 0.990:
        print("15@ " + str(tau) + ", " + str(fid_sym))

    update_plots(fig, ax, \
        [p1, p3, p15], \
        [p_t, p_t, p_t], \
        [sym1, sym3, sym15] )

np.savetxt("data/figfindTaus", taus)
np.savetxt("data/figfind1.txt", sym1)
np.savetxt("data/figfind3.txt", sym3)
np.savetxt("data/figfind15.txt", sym15)
fig.savefig("data/figfind.png")

print("Done! Press Enter to exit.")
raw_input()
