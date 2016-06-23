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
        return ((-1) ** sumHeavisideMonotonic(t, ts) ) * eta_0
    return eta

def ezGenerateEta(t_end, tau_c, eta_0):
    return generateEta(generateJumpTimes(t_end, tau_c), eta_0)

# Can use takewhile over filter because monotonic
def sumHeavisideMonotonic(t, ts):
    return len([x for x in takewhile(lambda t_: t_ <= t,ts)])

# Eq. 8
def generateRho(rho_0, N, Us):
    def rho(t):
        U_k_ts = [np.matrix(U_k(t)) for U_k in Us]
        terms = [U_k_t * rho_0 * U_k_t.H for U_k_t in U_k_ts]
        return (1./N) * sum(terms)
    return rho

# Eq. 9
# Generate unitary time evolutions
# Ignore time-ordering for now...
def generateU_k(a, eta_k):
    def U_k(t):
        a_, err1 = integrate.quad(a, 0, t)
        eta_k_, err2 = integrate.quad(eta_k, 0, t)
        H_ = -(1.j/hbar) * H_t(a_, eta_k_)
        return expm(np.array(H_))
    return U_k

# Generate a rho
def ezGenerate_Rho(a, t_end, tau_c, eta_0, rho_0, N):
    Us = [ezGenerateU_k(a, t_end, tau_c, eta_0) for i in range(0,N)]
    return generateRho(rho_0, N, Us)

# Generate a U_k
def ezGenerateU_k(a, t_end, tau_c, eta_0):
    return generateU_k(a, ezGenerateEta(t_end, tau_c, eta_0))
    # return generateU_k(a, eta_sys)

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


####

# Bit flip on computational basis
rho_0 = dm_1
rho_f = dm_0
eta_0 = Delta
N = 1000 # number of RTN trajectories
t_end = 32 / hoa # end of RTN

tau_c_0 = 0.2 / hoa
tau_c_f = 30. / hoa
dtau_c = 1.2 / hoa
tau_c = tau_c_0
tau_cs = [tau_c]
while tau_c < tau_c_f:
    tau_c += dtau_c
    tau_cs.append(tau_c)

fids_pi = []
fids_C = []
fids_SC = []

start = time.time()
for i in range(len(tau_cs)):
    print(str(i) + "/" + str(len(tau_cs)))
    tau_c = tau_cs[i]

    rho_pi = ezGenerate_Rho(a_pi, t_end, tau_c, eta_0, rho_0, N)
    fid_pi = fidSingleTxDirect(rho_f, rho_pi, T_pi)
    fids_pi.append(fid_pi)

    rho_C = ezGenerate_Rho(a_C, t_end, tau_c, eta_0, rho_0, N)
    fid_C = fidSingleTxDirect(rho_f, rho_C, T_C)
    fids_C.append(fid_C)

    # rho_SC = ezGenerate_Rho(a_SC, t_end, tau_c, eta_0, rho_0, N)
    # fid_SC = fidSingleTxDirect(rho_f, rho_SC, T_SC)
    # fids_SC.append(fid_SC)
print("time taken: " + str(time.time() - start))

fig = plt.figure()
plt.plot(tau_cs, fids_pi, 'b--', label="pi pulse")
plt.plot(tau_cs, fids_C, 'r-', label="CORPSE pulse")
# plt.plot(tau_cs, fids_SC, 'r--', label="SCORPSE pulse")
# plt.axis([0, 30, 0.975, 1])
plt.legend(loc='best')
plt.show()
