import sys
from numpy import *
from scipy import integrate
from scipy.constants import hbar, pi
import pylab as p

# electron relaxation time between singlet triplet state
# s. pasini paper referenced t in units of "tau_p"??
tau = 1.

# Delta is arbitrary for now
# D = 1e12    # Delta
D = 6e-33

i_ = 1.0j
def iw(t):
    return i_*w*t

def normalize(v):
    magnitude = sqrt(sum(map(lambda x: x**2, v)))
    print(magnitude)
    return map(lambda v_i: float(v_i)/magnitude, v)

# Makes X(t) driving pulse function
# theta is either pi or pi/2
# a, b are constants
def X_factory(theta, a, b, antisym):
    def X_sym(t):
        _1 = theta / 2
        _2 = (a - _1) * cos((2 * pi  / tau) * t)
        _3 = a * cos((4 * pi  / tau) * t)
        return _1 + _2 - _3

    def X_antisym(t):
        _1 = X_sym(t)
        _2 = b * sin((2 * pi  / tau) * t)
        _3 = (b/2) * sin((4 * pi  / tau) * t)
        return _1 + _2 - _3

    if antisym:
        return X_antisym
    return X_sym



def dChis_dt_factory(X):
    # wraps the coupled diff eqs.
    def dChis_dt(t, Chi):
        a = Chi[0]      # Chi_plus component
        b = Chi[1]      # Chi_minus component
        c = -1.0j/hbar

        def Chi_plus_dot_f(a, b):
            return D*a + (hbar / tau)*b*X(t)

        def Chi_minus_dot_f(a, b):
            return (hbar / tau)*a*X(t) - D*b

        a_dot = Chi_plus_dot_f(a, b)
        b_dot = Chi_minus_dot_f(a,b)

        return c*array([
                    a_dot,
                    b_dot,
                    ])

    return dChis_dt

theta1 = pi
#
theta2 = pi / 2

## Symmetric
a1_sym = -2.159224 * (1/tau)
#
a2_sym = -5.015588 * (1/tau)
# 'b' is unnecessary

## Antisymmetric
a1_asym = 5.263022 * (1/tau)
b1_asym = 17.850535 * (1/tau)

a2_asym = -16.809353 * (1/tau)
b2_asym = 15.634390 * (1/tau)

# Symmetric
X_s_f = X_factory(theta1, a1_sym, None, False)
dChiSym_dt = dChis_dt_factory(X_s_f)

X_s_f = X_factory(theta1, a2_sym, None, False)
dChiSym2_dt = dChis_dt_factory(X_s_f)

# Antisymmetric
X_a_f = X_factory(theta1, a1_asym, b1_asym, True)
dChiAsym_dt = dChis_dt_factory(X_a_f)
#
X_a2_f = X_factory(theta1, a2_asym, b2_asym, True)
dChiAsym2_dt = dChis_dt_factory(X_a2_f)

# Setup orthogonal spinors
# Assume Chi_0 eual superposition of Computational Basis 0 and 1
# At t=0, H = D*sigma_z + 0 since X(0) = 0.
# Thus H(0) eigenspinors are same as sigma_z = Z.
# eigenspinors are typical (1 0)T and (0 1)T
cb0 = array([1., 0.])
cb1 = array([0., 1.])

Chi_0 = normalize(cb0 + cb1)
print(Chi_0)
# Chi_0 = Chi_plus_0 + Chi_minus_0

# Integration "params"
steps = 800.
t0 = 0.
t1 = tau
dt = t1/steps     # 100 steps

def wrapIntegrate(func, init, init_t):
    vals = []
    sol = integrate.ode(func).set_integrator('zvode')
    sol.set_initial_value(init, init_t)
    while sol.successful() and sol.t < t1:
        t_ = sol.t + dt
        res = sol.integrate(t_)
        vals.append(res)
    return (vals, sol.successful())

ChiS_, success = wrapIntegrate(dChiSym_dt, Chi_0, t0)
ChiA_, success = wrapIntegrate(dChiAsym_dt, Chi_0, t0)

if not success:
    print("Failure?")
    sys.exit()
else:
    print("Success?")

ChiS = array(ChiS_)
ChiA = array(ChiA_)
ts = linspace(t0, t1, len(ChiS))          # time as x coordinate

ChiS_p, ChiS_m = ChiS.T                        # Transverse
ChiA_p, ChiA_m = ChiA.T                        # Transverse

def sq(x):
    return x**2

def norm(x):
    return sqrt(dot(x, conjugate(x)))

def decorate(p, title, xlabel="time", ylabel="values"):
    p.legend(loc='best')
    p.xlabel(xlabel)
    p.ylabel(ylabel)
    p.title(title)

## Plotting
delta_str = "Delta = " + str(D)
sym_title = "Chi Symmetric; Real values; " + delta_str
asym_title = "Chi Antisymmetric; Real values; " + delta_str

f1 = p.figure(1)
p.plot(ts, ChiS_p, 'r-', label='Chi_+')
p.plot(ts, ChiS_m, 'b-', label='Chi_-')
decorate(p, sym_title)

f2 = p.figure(2)
fidS = map(norm, ChiS_p)
p.plot(ts, fidS, 'r-', label='Fidelity = |<Chi|0>')
decorate(p, sym_title)
print("Max Fidelity Sym: " + str(max(fidS)))

f1 = p.figure(3)
p.plot(ts, ChiA_p, 'r-', label='Chi_+')
p.plot(ts, ChiA_m, 'b-', label='Chi_-')
decorate(p, asym_title)

f2 = p.figure(4)
fidA = map(norm, ChiA_p)
p.plot(ts, fidA, 'r-', label='Fidelity = |<Chi|0>')
decorate(p, asym_title)
print("Max Fidelity Antisym: " + str(max(fidA)))

### Fidelity vs Theta

do_s = False
do_a = True

thetas = linspace(0., pi, 42)
max_fidsS = []
avg_fidsS = []
max_fidsA = []
avg_fidsA = []

for theta in thetas:
    #Sym
    if do_s:
        X_s_f = X_factory(theta, a1_asym, None, False)
        dChiSym_dt = dChis_dt_factory(X_s_f)
        ChiA_, success = wrapIntegrate(dChiSym_dt, Chi_0, t0)
        ChiS_p, ChiS_m = ChiS.T
        fidsS = map(norm, ChiS_p)
        max_fidsS.append(max(fidsS))
        avg_fidsS.append(average(fidsS))
    if do_a:
    #Asym
        X_a_f = X_factory(theta, a1_asym, b1_asym, True)
        dChiAsym_dt = dChis_dt_factory(X_a_f)
        ChiA_, success = wrapIntegrate(dChiAsym_dt, Chi_0, t0)
        ChiA_p, ChiA_m = ChiA.T
        fidsA = map(norm, ChiA_p)
        max_fidsA.append(max(fidsA))
        avg_fidsA.append(average(fidsA))

fz = p.figure(5)
if do_s:
    p.plot(thetas, max_fidsS, 'r-', label='Max Fidelity S')
    p.plot(thetas, avg_fidsS, 'r--', label='Average Fidelity S')
if do_a:
    p.plot(thetas, max_fidsA, 'b-', label='Max Fidelity A')
    p.plot(thetas, avg_fidsA, 'b--', label='Average Fidelity A')
decorate(p, "Max/Avg fidelity vs theta", xlabel="Theta", ylabel="Fidelity")

p.show()
