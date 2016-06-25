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
from itertools import takewhile, repeat
from multiprocess import Pool
from parallel import parallel
import linmax
import dill
import pdb; pdb.set_trace

cpus = 4
pool = Pool(processes=cpus)

hbar = 1.
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

hoa = 1.

sigmaX = np.matrix([    [0., 1.]  ,
                        [1., 0.]  ])

sigmaY = np.matrix([    [0.,-1.j] ,
                        [1.j, 0.] ])

sigmaZ = np.matrix([    [1., 0.]  ,
                        [0.,-1.]  ])

# Eq. 15
# Pi pulse
T_pi = 1. * pi
def a_pi(t):
    if 0 < t and t < T_pi:
        return a_max
    return 0

# Eq. 16
# CORPSE
T_C = 13 * pi / 3
def a_C(t):
    if t < 0:
        return 0
    if t < (pi / 3):
        return a_max
    if t <= (2 * pi):
        return -a_max
    if t < T_C:
        return a_max
    return 0

# Eq. 17
# SCORPSE, i.e. Short CORPSE
T_SC = 7 * pi / 3
def a_SC(t):
    if t < 0:
        return 0
    if t < (pi / 3):
        return -a_max
    if t <= (2 * pi):
        return a_max
    if t < T_SC:
        return -a_max
    return 0

######
U_count = [0]
rho_time = [0.]
u_time = [0.]
eta_time = [0.]
# Eq. 2
def H_t(a_t, eta_t):
    return 0.5 * a_t * sigmaX + 0.5 * eta_t * sigmaZ
# Can use takewhile over filter because monotonic
# The heaviside function used is 1 at 0.
def sumHeavisideMonotonic(t, ts):
    return len([x for x in takewhile(lambda t_: t_ <= t,ts)])

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

class EtaProvider:
    def __init__(self, eta_0, t_end, tau_c):
        self.eta_0 = eta_0
        self.t_end = t_end
        self.tau_c = tau_c

    # Eq. 4
    # Jump times for RTN trajectory
    # t_end can be understood in units of tau_c
    # tau_c is the Noise correlation time
    def generateJumpTimes(self):
        ts = []
        t = 0.
        while t < self.t_end:
            p = np.random.random_sample()
            dt = (-self.tau_c) * np.log(p)
            t += dt
            ts.append(t)
        return ts

    # Eq. 5
    # Generate a noise function from jumps
    # i.e. an RTN trajectory
    def get(self):
        s = time.time()
        ts = self.generateJumpTimes()
        eta_time[0] += (time.time() - s)
        return Eta(self.eta_0, ts)

class Eta():
    def __init__(self, eta_0, ts):
        self.eta_0 = eta_0
        self.ts = ts
    def at(self, t):
        s = time.time()
        if t == 0:
            return self.eta_0
        res = ((-1) ** sumHeavisideMonotonic(t, self.ts)) * self.eta_0
        eta_time[0] += (time.time() - s)
        return res

class UkProvider():
    def __init__(self, pulse, steps, etaProvider):
        self.pulse = pulse
        self.steps = steps
        self.etaProvider = etaProvider

    def get(self):
        def U_k(t):
            start = time.time()
            eta = self.etaProvider.get().at
            def G(t_):
                return np.array(-(1.j/hbar) * H_t(self.pulse(t_), eta(t_)))
            steps = 420
            ts = np.linspace(0., t, steps)
            dt = ts[1] - ts[0]
            Ss = [G(t) for t in ts]
            C = reduce(BCH, Ss)
            U_count[0] += 1

            end = time.time()
            delts = str(end - start)
            sys.stdout.write("\rU++: " + str(U_count[0]) + "; " + delts)
            sys.stdout.flush()

            return expm(C)
        return U_k

class Uk():
    def __init__(self, G, steps):
        self.G = G
        self.steps = steps

    def at(self, t):
        start = time.time()

        steps = self.steps
        ts = np.linspace(0., t, steps)
        dt = ts[1] - ts[0]
        Ss = [self.G(t) for t in ts]
        C = reduce(BCH, Ss)
        # U_count[0] += 1
        # end = time.time()
        # delts = str(end - start)
        # sys.stdout.write("\rU++: " + str(U_count[0]) + "; " + delts)
        # sys.stdout.flush()
        # u_time[0] += (end - start)
        return linmax.powerexp(C)

class RhoFactory():
    def __init__(self, rho_0, N, uSteps, etaProvider):
        self.rho_0 = rho_0
        self.N = N
        self.uSteps = uSteps
        self.etaProvider = etaProvider

    def make(self, pulse, pulse_end_time):
        ukProvider = UkProvider(pulse, self.uSteps, etaProvider)
        return Rho(self.rho_0, self.N, pulse_end_time, ukProvider)

class Rho():
    def __init__(self, rho_0, N, pulse_end_time, ukProvider):
        self.rho_0 = rho_0
        self.N = N
        self.pulse_end_time = pulse_end_time
        self.ukProvider = ukProvider

    def at(self, t):
        # Uk_ts = [np.matrix(self.ukProvider.get()(t)) for i in range(N)]

        U_k_ts = pool.map(lambda i: np.matrix(self.ukProvider.get()(t)), range(N))
        s = time.time()
        terms = [U * self.rho_0 * U.H for U in Uk_ts]
        rho_time[0] += time.time() - s
        return (1./self.N) * sum(terms)

    def fidelity(self, rho_f):
        return np.trace(rho_f.H * self.at(self.pulse_end_time))


######


# Bit flip on computational basis
rho_0 = dm_1
rho_f = dm_0
eta_0 = Delta
uSteps = 100 # number of samplings of Hamiltonian
N = 10 # number of RTN trajectories
t_end = 31 * hoa # end of RTN

tau_c_0 = 0.4 * hoa
tau_c_f = 30. * hoa
dtau_c = .5 * hoa

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
prev_time = -1
for i in range(len(tau_cs)):
    ministart = time.time()
    # if i % 15 is 0:
    sys.stdout.write("\r"+str(i) + "/" + str(len(tau_cs)) + "  " + str(prev_time))
    sys.stdout.flush()
    U_count[0] = 0
    print("\n")

    etaProvider = EtaProvider(Delta, t_end, tau_c)
    rhoFactory = RhoFactory(rho_0, N, uSteps, etaProvider)

    fids_pi.append(rhoFactory.make(a_pi, T_pi).fidelity(rho_f))

    fids_C.append(rhoFactory.make(a_C, T_C).fidelity(rho_f))

    fids_SC.append(rhoFactory.make(a_SC, T_SC).fidelity(rho_f))

    miniend = time.time()
    prev_time = miniend - ministart
print("time taken: " + str(time.time() - start))
print("""
eta_time {eta_time[0]}
u_time {u_time[0]}
rho_time {rho_time[0]}
""".format(**locals()))
