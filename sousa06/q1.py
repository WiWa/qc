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
from itertools import takewhile

###
# Reproduction of:
# High-fidelity one-qubit operations under random telegraph noise
# Mikko Mottonen and Rogerio de Sousa
###

sigmaX = np.matrix([    [0, 1]  ,
                        [1, 0]  ])

sigmaY = np.matrix([    [0,-1j] ,
                        [1j, 0] ])

sigmaZ = np.matrix([    [1, 0]  ,
                        [0,-1]  ])


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

# Eq. 15
# Pi pulse
def a_pi(t):
    if 0 < t and t < (pi * hbar / a_max):
        return a_max
    return 0

# Eq. 16
# CORPSE
def a_C(t):
    if t < 0:
        return 0
    if t < (pi / 3):
        return a_max
    if t <= (2 * pi):
        return -a_max
    if t < (13 * pi / 3):
        return a_max
    return 0

# Eq. 17
# SCORPSE, i.e. Short CORPSE
def a_SC(t):
    if t < 0:
        return 0
    if t < (pi / 3):
        return -a_max
    if t <= (2 * pi):
        return a_max
    if t < (7 * pi / 3):
        return -a_max
    return 0

# Systematic Error
def eta_sys(t):
    return Delta


# Eq. 2
# Hamiltonian
def generateH(a, eta):
    def H_(t):
        return 0.5 * a(t) * sigmaX + 0.5 * eta(t) * sigmaZ
    return H_

# Evaluate H given pre-evaluated a(t) and eta(t)
def H_t(a_t, eta_t):
    return 0.5 * a_t * sigmaX + 0.5 * eta_t * sigmaZ

# Eq. 4
# Jump times for RTN trajectory
# t_end can be understood in units of tau_c
# tau_c is the Noise correlation time
def generateJumpTimes(t_end, tau_c):
    ts = []
    t = 0.
    while t < t_end:
        p = np.random.random_sample()
        dt = -tau_c * np.log(p)
        t += dt
        ts.append(t)
    return ts

# Eq. 5
# Generate a noise function from jumps
# i.e. an RTN trajectory
# The heaviside function used is 1 at 0.
def generateEta(ts, eta_0):
    def eta(t):
        return (-1. ** sumHeavisideMonotonic(t, ts) ) * eta_0
    return eta

# Can use takewhile over filter because monotonic
def sumHeavisideMonotonic(t, ts):
    return len([x for x in takewhile(lambda t_: t_ <= t,ts)])

# Eq. 8
def generateRho(rho_0, N, Us):
    def rho(t):
        terms = []
        for k in range(0, N):
            U_k_t = np.matrix(Us[k](t))
            terms.append(U_k_t * rho_0 * U_k_t.H)
        return (1./N) * sum(terms)
    return rho

# Eq. 9
# Generate unitary time evolutions
# Ignore time-ordering for now...
def generateU_k(a, eta_k):
    # steps = 1000
    # H = generateH(a, eta_k)
    # def U_k(t):
    #     H_ = -(1.j/hbar) * integrateM2(H, 0, t, steps)
    #     return exmp(H_)
    # return U_k
    def U_k(t):
        a_, err1 = integrate.quad(a, 0, t)
        eta_k_, err2 = integrate.quad(eta_k, 0, t)
        H_ = -(1.j/hbar) * H_t(a_, eta_k_)
        return expm(H_)
    return U_k

# Generate a rho
def ezGenerate_Rho(a, t_end, tau_c, eta_0, rho_0, N):
    Us = [ezGenerateU_k(a, t_end, tau_c, eta_0) for i in range(0,N)]
    return generateRho(rho_0, N, Us)

# Generate a U_k
def ezGenerateU_k(a, t_end, tau_c, eta_0):
    return  generateU_k(a,
            generateEta(
            generateJumpTimes(t_end, tau_c), eta_0))

# Integrate a 2x2 matrix
# def integrateM2(M, t_0, t_f):
#     row0 = [ integrate.quad(f, t_0, t_f) for f in M[0] ]
#     row1 = [ integrate.quad(f, t_0, t_f) for f in M[1] ]
#     return np.matrix([ row0, row1])
# dumb implementation for now
def integrateM2(M, t_0, t_f, steps):
    t_ = t_0
    M_ = M(t_0)
    dt = ( t_f - t_0 ) / float(steps)
    while t_ <= t_f:
        t_ += dt
        M_ += M(t_) * dt
    return M_

# Fidelity of a single transformation rho_0 -> rho_f
def fixSingleTx(rho_0, N, Us, T, rho_f):
    rho = generateRho(rho_0, N, Us)
    return fidSingleTxDirect(rho_f, rho, T)

# Eq. 10
# Fidelity of a single transformation rho_0 -> rho_f
def fidSingleTxDirect(rho_f, rho, T):
    return np.trace(rho_f.H * rho(T))

print(generateJumpTimes(10, 1))
print(sumHeavisideMonotonic(3, [1,2,3,4,5,6]))
H_ = generateH(lambda t: 1.5*t, lambda t: 3.5*t)
print(H_(1))
print(H_(0))

T = 1.
start = time.time()
G = integrateM2(H_, 0., T, 1000)
print("time taken: " + str(time.time() - start))
print(G)

####

# Bit flip on computational basis
rho_0 = dm_1
rho_f = dm_0
eta_0 = Delta
N = 420 # number of RTN trajectories
t_end = pi
tau_c = 1
rho_pi = ezGenerate_Rho(a_pi, t_end, tau_c, eta_0, rho_0, N)
rho = rho_pi
# print(rho_f)
fid = fidSingleTxDirect(rho_f, rho, t_end)
print(fid)
