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
parallel = (not profiling) and True


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
def generateEta(ts, eta_0):
    def eta(t):
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
def generateU_k(a, eta_k, stepsize=0.03, t0=0.):
    def U_k(t):
        # if t0 <= t and t <= :

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
            return dt * -(1.j) * H_t2(a(t_), eta_k(t_))

        # C = np.array([[1,0],[0,1]])
        # Ss = [G(t) for t in ts]
        # C = reduce(BCH, Ss)
        C = G(ts[0])
        for i in range(1, len(ts)):
            # G_avg = 0.5 * (G(ts[i-1]) + G(ts[i]))
            C = BCH(C, G(ts[i]))


        # C = sum(Ss)
        # U_count[0] += 1

        # print(C)
        end = time.time()
        delts = str(end - start)
        # sys.stdout.write("\rU++: " + str(U_count[0]) + "; " + delts)
        # sys.stdout.flush()
        # return linmax.powerexp(C)
        return expm(C)
    return U_k

def stepForwardMats(G, t_0, t, steps):
    ts = np.linspace(t_0, t, steps)
    dt = ts[1] - ts[0]
    Ss = [dt*G(t_) for t_ in ts]
    return Ss

k1 = np.array(1/2.,np.complex128)
k2 = np.array(1/12.,np.complex128)
k3 = np.array(-1/12.,np.complex128)
k4 = np.array(-1/24.,np.complex128)
k5 = np.array(-1/720.,np.complex128)
# Baker-Campbell-Hausdorff approx
def BCH(A, B, o5=False):
    start = time.time()

    c1 = np.dot(k1,comm(A, B))
    c2 = np.dot(k2,comm(A, c1))
    c3 = np.dot(k3,comm(B, c1))
    c4 = np.dot(k4,comm(B, c2))
    res = A + B + c1 + c2 + c3 + c4
    if o5:
        c5 = np.dot(k5,(bch5(A, B) + bch5(B, A)))
        res += c5
    if profiling:
        bch_time[0] += time.time() - start
    return res

def bch5(A, B):
    return comm(B,comm(B,comm(B,comm(B,A))))

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
        Us_nm_k = U_k[m:(n-1)]
        # Us_nm_k.reverse()
        Us_m1_k = U_k[0:(m-1)]
        # Us_m1_k.reverse()

        U_nm_k = reduce(np.dot, reversed(Us_nm_k),identity)
        U_nm_kH = U_nm_k.conj().T
        U_m1_k = reduce(np.dot, reversed(Us_m1_k),identity)
        U_m1_kH = U_m1_k.conj().T

        lambda_mk = reduce(np.dot, [U_nm_kH, rho_f, U_nm_k])
        lambda_H = lambda_mk.conj().T
        rho_mk = reduce(np.dot, [U_m1_k, rho_0, U_m1_kH])
        term = np.trace(np.dot(lambda_H, comm(sigmaX, rho_mk)))
        return term
    if parallel:
        terms = pool.map(makeTerm, range(N))
    else:
        terms = map(makeTerm, range(N))
    res = sum(terms)
    res = c * res
    return res

def GRAPE(T, n, N, rho_0, rho_f, tau_c, eta_0, stepsize, amps, epsilon=0.01):
    pulses = buildGrapePulses(amps, T)
    ts = np.linspace(0., T, n+1)
    Us_k = [] # Every U_k is actually a list of U_m
    new_amps = []
    def genUs_m(m):
        t0 = ts[m]
        te = ts[m+1]
        t_avg = (t0 + te) / 2.
        U_m = ezGenerateU_k(pulses[m], te, tau_c, eta_0, stepsize)(t_avg)
        return U_m
    for k in range(N):
        if not profiling and parallel:
            Us_m = pool.map(genUs_m, range(n))
        else:
            Us_m = map(genUs_m, range(n))
        Us_k.append(Us_m)
    for m in range(n):
        a_m = amps[m]
        d_phi_d_am = grad_phi_am(T, N, m, Us_k, rho_0, rho_f)
        # Eq. 13
        new_amp = a_m + epsilon*a_max*d_phi_d_am
        new_amp = max(new_amp, -a_max)
        new_amp = min(new_amp, a_max)
        new_amps.append(new_amp)
    return new_amps


def pulseMaker(t0, te, amp):
    def pulse(t):
        if t0 < t and t <= te:
            return amp
        return 0.
    return pulse

def buildGrapePulses(amps, T):
    n = len(amps)
    ts = np.linspace(0., T, n + 1)
    pulses = []
    for m in range(1, len(ts)):
        t0 = ts[m-1]
        te = ts[m]
        amp = amps[m-1]
        pulse = pulseMaker(t0, te, amp)
        pulses.append(pulse)
    return pulses

def aggPulse(pulses):
    def pulse(t):
        return sum([pulse_section(t) for pulse_section in pulses])
    return pulse

####

# Bit flip on computational basis
rho_0 = dm_1
rho_f = dm_0
eta_0 = Delta

tau_c_0 = 0.2 * hoa
tau_c_f = 16. * hoa
###
# Performance Params
###
dtau_c = 0.42 * hoa
N = 400 # number of RTN trajectories
stepsize = 0.03 # Step-forward matrices step size

T_G = 4 * hoa # sousa figure 2
n = 4 # number of different pulse amplitudes
epsilon = 0.1 # amount each gradient step can influence amps
grape_steps = 10 # number of optimization steps
###
t_end = tau_c_f + 0.42 * hoa # end of RTN

profiling = False

tau_c = tau_c_0
tau_cs = [tau_c]
while tau_c < tau_c_f:
    tau_c += dtau_c
    tau_cs.append(tau_c)

# tau_cs = [0.4, 1.8, 3., 5., 10.]
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

fids_pi = []
fids_C = []
fids_SC = []
fids_G = []

np.random.seed()
cpus = 8
if not profiling and parallel:
    print("POOL")
    pool = Pool(processes=cpus)

def ezmap(f, xs):
    if parallel:
        return pool.map(f, xs)
    return map(f, xs)

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
    js = generateJumpTimes(t_end, tau_c)

    rho_pi, Us = ezGenerate_Rho(a_pi, t_end, tau_c, eta_0, rho_0, N, stepsize)
    fid_pi = fidSingleTxDirect(rho_f, rho_pi, T_pi)
    fids_pi.append(fid_pi)

    rho_C, Us = ezGenerate_Rho(a_C, t_end, tau_c, eta_0, rho_0, N, stepsize)
    fid_C = fidSingleTxDirect(rho_f, rho_C, T_C)
    fids_C.append(fid_C)

    rho_SC, us = ezGenerate_Rho(a_SC, t_end, tau_c, eta_0, rho_0, N, stepsize)
    fid_SC = fidSingleTxDirect(rho_f, rho_SC, T_SC)
    fids_SC.append(fid_SC)

    ### Grape
    init_amps = [0 for i in range(n)]
    grape_amps = init_amps
    for i in range(grape_steps):
        grape_amps = GRAPE(T_G, n, N, rho_0, rho_f, tau_c, eta_0, stepsize, grape_amps, epsilon)
    grape_pulse = aggPulse(buildGrapePulses(grape_amps, T_G))
    rho_G, Us = ezGenerate_Rho(grape_pulse, t_end, tau_c, eta_0, rho_0, N, stepsize)
    fid_G = fidSingleTxDirect(rho_f, rho_G, T_G)
    fids_G.append(fid_G)

    miniend = time.time()
    prev_time = miniend - ministart
print("time taken: " + str(time.time() - start))

fig = plt.figure()

np.savetxt("data/fids_pi.txt", fids_pi)
np.savetxt("data/fids_C.txt", fids_C)
np.savetxt("data/fids_SC.txt", fids_SC)

xnew = np.linspace(tau_cs[0],tau_cs[-1],100)
fids_pi = spline(tau_cs, fids_pi, xnew)
fids_C = spline(tau_cs, fids_C, xnew)
fids_SC = spline(tau_cs, fids_SC, xnew)

plt.plot(xnew, fids_pi, 'b--', label="pi pulse")
plt.plot(xnew, fids_C, 'r-', label="CORPSE pulse")
plt.plot(xnew, fids_SC, 'r--', label="SCORPSE pulse")
# plt.axis([0, 30, 0.975, 1])
plt.xlabel("tau_c / (hbar / a_max)")
plt.ylabel("fidelity \\phi(rho_f, rho_0)")
plt.legend(loc='best')

# smooth_ax = p.axes([0.8, 0.025, 0.1, 0.04])
# smooth_btn = Button(smooth_ax, 'Smooth', color=axcolor, hovercolor='0.975')
# smooth_btn.on_clicked(smooth)

plt.show()
