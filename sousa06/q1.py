import os, sys, time
sys.path.append("/home/arbiter/qc")

from mod import (
                np, sp,
                mpl, plt,
                qc_dir,
                unpy,
                )
from scipy import integrate
from scipy.constants import hbar, pi
from scipy.linalg import expm
from itertools import takewhile, repeat

###
# Reproduction of:
# High-fidelity one-qubit operations under random telegraph noise
# Mikko Mottonen and Rogerio de Sousa
###

sigmaX = np.matrix([    [0., 1.]  ,
                        [1., 0.]  ])

sigmaY = np.matrix([    [0.,-1.j] ,
                        [1.j, 0.] ])

sigmaZ = np.matrix([    [1., 0.]  ,
                        [0.,-1.]  ])


####

# Maximum amplitude of control field
# set to hbar so that hbar/a_max = 1
a_max = hbar

# Noise strength
# 0.125 is the factor used in the paper
Delta = 0.125 * a_max

# Computational Bases
cb_0 = np.array([1, 0])
cb_1 = np.array([0, 1])

# Density Matrices
dm_0 = np.matrix([ cb_0, [0, 0] ]).T
dm_1 = np.matrix([ [0, 0], cb_1 ]).T

### Pulses

hoa = (hbar / a_max)

# Eq. 15
# Pi pulse
T_pi = pi * hoa
def a_pi(t):
    if 0 < t and t < T_pi:
        return a_max
    return 0

# Eq. 16
# CORPSE
T_C = (13 * pi / 3) * hoa
def a_C(t):
    if t < 0:
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
    if t < 0:
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
    return np.matrix(U_k(t))
def generateRho(rho_0, N, Us):
    def rho(t):
        U_k_ts = [np.matrix(U_k(t)) for U_k in Us]
        # dec_Us = zip(Us, repeat(t))

        # U_k_ts = pool.map(evalMat, dec_Us)

        # U_k_ts = pool.map(evalMat, Us)
        terms = [U_k_t * rho_0 * U_k_t.H for U_k_t in U_k_ts]
        return (1./N) * sum(terms)
    return rho

# Eq. 9
# Generate unitary time evolution
# Ignore time-ordering for now...
def generateU_k(a, eta_k, js):
    def U_k(t):
        # a_t, err1 = integrate.quad(a, 0, t)
        # eta_k_t, err2 = integrate.quad(eta_k, 0, t)
        # H_ = -(1.j/hbar) * H_t(a_t, eta_k_t)
        # return expm(np.matrix(H_))
        def G(t_):
            return -(1.j/hbar) * H_t(a(t_), eta_k(t_))
        Ss = stepForwardMats(G, 0., t, 100)
        C = reduce(BCH, Ss)
        # print(H_)
        return expm(C)

        # e_eta = expm(np.matrix(H_eta(eta_k_t)))
        # A = np.matrix(-(1.j/hbar) * H_a(a_t))
        # B = np.matrix(-(1.j/hbar) * H_eta(eta_k_t))
        # C = BCH(A, B)
        # print(C - A - B)
        # return expm(np.matrix(C))
    return U_k

def stepForwardMats(G, t_0, t, steps):
    ts = np.linspace(t_0, t, steps)
    dt = ts[1] - ts[0]
    Ss = [dt*G(t_) for t_ in ts]
    return Ss

def posDyson(a, eta, ts, t):
    time_order = filter(lambda t_: t_ < t, ts)
    time_order.reverse()
    H_ = np.matrix([[1., 0.], [0., 1.]])
    for i in range(1, len(time_order)):
        t_prev = time_order[i - 1]
        t_next = time_order[i]
        a_, err2 = integrate.quad(a, t_prev, t_next)
        eta_, err2 = integrate.quad(eta, t_prev, t_next)
        H_ *= np.matrix(H_a(a_)) + np.matrix(H_eta(eta_))
    return H_

# Baker-Campbell-Hausdorff approx
def BCH(A, B):
    c1 = (1/2.)*comm(A, B)
    c2 = (1/12.)*comm(A, c1)
    c3 = -(1/12.)*comm(B, c1)
    c4 = -(1/24.)*comm(B, c2)
    return A + B + c1 + c2 + c3 + c4

# Commutator
def comm(A, B):
    return (A * B) - (B * A)
# js = generateJumpTimes(15, 1)
# e = generateEta(js, 1)
# print([e(t/10.) for t in range(0, 110)])
# et = integrate.quad(e, 0, 10)
# print(et)

# Generate a rho
def ezGenerate_Rho(a, t_end, tau_c, eta_0, rho_0, N):
    Us = [ezGenerateU_k(a, t_end, tau_c, eta_0) for i in range(N)]
    return generateRho(rho_0, N, Us)

# Generate a U_k
def ezGenerateU_k(a, t_end, tau_c, eta_0):
    js = generateJumpTimes(t_end, tau_c)
    return generateU_k(a, generateEta(js, eta_0), js)
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
    return np.trace(rho_f.H * rho(T))

# Eq. 11
def fidSingleTxFull(rho_0, rho_f, T, N, Us):
    U_k_ts = [np.matrix(U_k(T)) for U_k in Us]
    mats = [rho_f.H * U_k_t * rho_0 * U_k_t.H for U_k_t in U_k_ts]
    terms = [np.trace(mat) for mat in mats]
    return (1./N) * sum(terms)


####

# Bit flip on computational basis
rho_0 = dm_1
rho_f = dm_0
eta_0 = Delta
N = 500 # number of RTN trajectories
t_end = 17 * hoa # end of RTN

tau_c_0 = 0.3 * hoa
tau_c_f = 16. * hoa
dtau_c = 0.7 * hoa
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
end of RTN:
    {t_end}
tau_c going from {tau_c_0} to {tau_c_f}
step size of {dtau_c}

Starting...
""".format(**locals()))

fids_pi = []
fids_C = []
fids_SC = []

start = time.time()
for i in range(len(tau_cs)):
    # if i % 15 is 0:
    sys.stdout.write("\r"+str(i) + "/" + str(len(tau_cs)))
    sys.stdout.flush()

    tau_c = tau_cs[i]

    rho_pi = ezGenerate_Rho(a_pi, t_end, tau_c, eta_0, rho_0, N)
    fid_pi = fidSingleTxDirect(rho_f, rho_pi, T_pi)
    fids_pi.append(fid_pi)

    rho_C = ezGenerate_Rho(a_C, t_end, tau_c, eta_0, rho_0, N)
    fid_C = fidSingleTxDirect(rho_f, rho_C, T_C)
    fids_C.append(fid_C)

    # Us = [ezGenerateU_k(a_C, t_end, tau_c, eta_0) for i in range(N)]
    # fids_C.append(fidSingleTxFull(rho_0,rho_f,T_C, N,Us))

    rho_SC = ezGenerate_Rho(a_SC, t_end, tau_c, eta_0, rho_0, N)
    fid_SC = fidSingleTxDirect(rho_f, rho_SC, T_SC)
    fids_SC.append(fid_SC)
print("time taken: " + str(time.time() - start))

fig = plt.figure()
plt.plot(tau_cs, fids_pi, 'b--', label="pi pulse")
plt.plot(tau_cs, fids_C, 'r-', label="CORPSE pulse")
plt.plot(tau_cs, fids_SC, 'r--', label="SCORPSE pulse")
# plt.axis([0, 30, 0.975, 1])
plt.xlabel("tau_c / (hbar / a_max)")
plt.ylabel("fidelity \\phi(rho_f, rho_0)")
plt.legend(loc='best')
plt.show()
