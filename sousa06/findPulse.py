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

def donorm(underlying,s,e, normproc="simple"):

    ma, mi = minmax(underlying, s,e)
    maxdiff = abs(ma - mi)
    # print maxdiff
    def simple(t):
        return underlying(t) / max(abs(ma), abs(mi))
    def full(t):
        return (2*underlying(t) / (maxdiff)) - a_max - (2*mi/maxdiff)
    def capped(t):
        return minabs(underlying(t), a_max)
    normf = None
    if normproc == "none":
        normf = underlying
    if normproc == "simple":
        normf = simple
    if normproc == "full":
        normf = full
    if normproc == "capped":
        normf = capped
    if normf is None:
        raise Exception("bad normproc: " + normproc)
    def normfbounded(t):
        if t < s or t > e:
            return 0
        return normf(t)
    return normfbounded

###### Sym/Antisym pulses
# 2.0 for "normal"
# ~4.9 for "capped", 3tauc < .99; 15tauc ~ .98; (useless)
# ~7.9 for normed, even worse than capped
# tau = 4.9 * hoa # Electron relaxation time; idk the "real" value :)
# Makes X(t) driving pulse function
# theta is either pi or pi/2
# a, b are constants
# This pulse lasts a single period: 0 -> tau
def X_factory(theta, constPair, antisym, tau, normproc="simple", a=None, b=None):
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

# XXX here
# naming: data/[pulse]/params
# p2: period=2, x-w: vary width, full: normproc="full"
# base = "data/sawtooth/p2_x-w_full/"
wave = "x2"
ptitle = "p2-5_x-w_norm"
base = "data/"+wave+"/"+ptitle+"/"
_periods=2.5

widthlist = list(np.arange(0.4*np.pi, 1.8*pi, 0.045*pi)) # width
# widthlist = list(np.arange(3.5*np.pi, 5.01*pi, 0.1*pi)) # width
# width_period_list = list([(w,p) for p in periodlist for w in widthlist])

xlist = widthlist

print base
if not os.path.exists(base):
    print "Creating directory"
    os.makedirs(base)

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

times = [0.2* hoa, 3.0* hoa, 12.0* hoa]
###
# Performance Params
###
N = 850 # number of RTN trajectories
stepsize = 0.024 # Step-forward matrices step size, dont lower

###
t_end = times[2] + 0.32 # end of RTN

cpus = 8
if not profiling and parallel:
    print("POOL")
    pool = Pool(processes=cpus)

# XXX fourier
def sawtooth(width, periods=2):
    # normalize to -1 -> 1 for now
    def pulse(t):
        if t < 0:
            return 0
        if t > width * periods:
            return 0
        return 2*(t % width)/width - 1
    return pulse
def rectifier(width, periods=2):
    # 0 to 1, width can be in units of pi
    def pulse(t):
        if t < 0:
            return 0
        if t > width * periods:
            return 0
        return sin((t % width)*pi/width)
    return pulse
def x2p(width, periods=2):
    # >= 0
    def pulse(t):
        if t < 0:
            return 0
        if t > width * periods:
            return 0
        return ((t % width) - (width/2.0)) ** 2
        # return (((t % width) - (width/2.0)) ** 2)/(2*a_max) - a_max
    return pulse
def square(width, periods=2):
    # -1 to 1
    def pulse(t):
        if t < 0:
            return 0
        if t > width * periods:
            return 0
        if t % width < 0.5*width:
            return 1
        return -1
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

# tau_c_0 = 0.2 * hoa
# # tau_c_f = 15. * hoa
# times = [0.2* hoa, 3.0* hoa, 18.0* hoa]
# ###
# # Performance Params
# ###
# N = 1420 # number of RTN trajectories
# stepsize = 0.023 # Step-forward matrices step size, dont lower
#
# ###
# t_end = 18.42 # end of RTN
#
# cpus = 8
# if not profiling and parallel:
#     print("POOL")
#     pool = Pool(processes=cpus)
#
# tau_start = (3.5 * pi/ 3.0) * hoa
# tau_end = (15 * pi / 3.0) * hoa
# # tau_start = (80 * pi/ 3.0) * hoa
# # tau_end = (82 * pi / 3.0) * hoa
# dtau = 0.35 * hoa
# t_ = tau_start
# taus = []
# while t_ < tau_end:
#     taus.append(t_)
#     t_ += dtau
a_start = -3.0
a_end = 3.0
da = 0.05
alist = []
a_ = a_start
while a_ <= a_end:
    alist.append(a_)
    a_ += da
# sym_pis = [X_factory(pi, a1_sym, 0, False, tau=t_) for t_ in taus]
sym1 = []
sym3 = []
sym12 = []

eta_0_a_max = eta_0 / a_max
print("""
Starting...
""".format(**locals()))

def ezmap(f, xs):
    if parallel:
        return pool.map(f, xs)
    return map(f, xs)

p_t = []

# XXX SHAPE
pulseshape = rectifier(2.,periods=_periods)
# pulseshape = X_factory(pi, 2, True, 1, normproc="full")
tis = np.linspace(0, 6, 1000)
pulseshape_data = [pulseshape(ti) for ti in tis]
pshape = plt.figure()
plt.plot(tis, pulseshape_data, 'b-')
# plt.show()

plt.ion()
fig, ax = plt.subplots()
fig.suptitle(wave + ":" + ptitle)

p1, = plt.plot(p_t, sym1, 'b--', label="t=1")
p3, = plt.plot(p_t, sym3, 'r-', label="t=3")
p15, = plt.plot(p_t, sym12, 'g--', label="t=12")

plt.xlabel(r"width in $\hbar / a_max$")
plt.ylabel(r"$\phi(\rho_f, \rho_0)$")
plt.legend(loc='best')
plt.show()
plt.pause(0.0001)

def SCORPSEfac(partition):
    def a_SC(t):
        if t <= 0:
            return 0
        if t < ((pi / 3) * hoa):
            return -a_max
        if t <= 2 * pi * hoa:
            return a_max
        if t < (partition):
            return -a_max
        return 0
    return a_SC


start = time.time()
prev_time = -1
# xlist = [1,2,3,4] # periods
# XXX here
pulsefs = [x2p(x, periods=_periods) for x in xlist]

start = time.time()
fullstart = start
for i in range(len(xlist)):
    # tau = taus[i]
    # a_sym = alist[i]
    print r"%i/%i"%(i, len(xlist))
    end = time.time()
    print end - start
    start = end

    x = xlist[i]
    p_t.append(x)
    pulsef = pulsefs[i]
    pulse_end = x * _periods


    # tau = x
    # theta = pi
    # constpair = 1
    # antisym = True

    # pulsef = X_factory(theta, constpair, antisym, tau)
    # XXX XXX

    rho_pulse0, us0 = ezGenerate_Rho(pulsef, t_end, times[0], eta_0, rho_0, N, stepsize)
    rho_pulse1, us1 = ezGenerate_Rho(pulsef, t_end, times[1], eta_0, rho_0, N, stepsize)
    rho_pulse2, us2 = ezGenerate_Rho(pulsef, t_end, times[2], eta_0, rho_0, N, stepsize)
    fid_0 = fidSingleTxDirect(rho_f, rho_pulse0, pulse_end)
    fid_1 = fidSingleTxDirect(rho_f, rho_pulse1, pulse_end)
    fid_2 = fidSingleTxDirect(rho_f, rho_pulse2, pulse_end)
    sym1.append(fid_0)
    sym3.append(fid_1)
    sym12.append(fid_2)

    if fid_0 > 0.985:
        print("1@ " + str(x) + ", " + str(fid_0))
    if fid_1 > 0.980:
        print("3@ " + str(x) + ", " + str(fid_1))
    if fid_2 > 0.980:
        print("12@ " + str(x) + ", " + str(fid_2))

    update_plots(fig, ax, \
        [p1, p3, p15], \
        [p_t, p_t, p_t], \
        [sym1, sym3, sym12] )

print "Total Time: " + str(time.time() - fullstart)

if not os.path.exists(base):
    print "Creating directory (at end)"
    os.makedirs(base)

np.savetxt(base+"figfindTaus.txt", xlist)
np.savetxt(base+"figfind1.txt", sym1)
np.savetxt(base+"figfind3.txt", sym3)
np.savetxt(base+"figfind12.txt", sym12)
pshape.savefig(base+"pulseshape.png")
fig.savefig(base+"figfind.png")

print("Done! Press Enter to exit.")
raw_input()
