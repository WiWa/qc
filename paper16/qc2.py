import sys
sys.path.append("/home/arbiter/qc")

from numpy import *
from scipy import integrate, pi
import pylab as p

from mod import promptSaveFigs, mapBothPend

# electron relaxation time between singlet triplet state
# s. pasini paper referenced t in units of "tau_p"??
tau = 1

i_ = 1.0j
def iw(t):
    return i_*w*t

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

# tau = tau_p?
# Almost def by Eq.43 in S.Pasini.
theta1 = pi
theta2 = pi / 2

steps = 100.
t0 = 0.
t1 = 1.
dt = t1/steps     # 100 steps
ts = linspace(t0, t1, t1/dt)          # time as x coordinate

def v_evolve(vs):
    v_pi, v_pi2 = vs

    v_pi_ = []
    v_pi2_ = []

    # Evolve time t_
    t_ = t0
    while t_ < t1:
        t_ += dt

        v_ = v_pi(t_)
        v_pi_.append(v_)

        v_ = v_pi2(t_)
        v_pi2_.append(v_)

    v_pi_vals = array(v_pi_)
    v_pi2_vals = array(v_pi2_)
    return [v_pi_vals, v_pi2_vals]

def v_1_factory(a1, a2, b1, b2, antisym):
    v_pi = X_factory(theta1, a1, b1, antisym)
    v_pi2 = X_factory(theta2, a2, b2, antisym)
    return [v_pi, v_pi2]


# the (1/tau) is mostly for aesthetic effect;
# tau = 1 because we're measuring in units of tau

# I'm pretty sure we're supposed to vary 'a'
# but let's start with what's given as the "answer".
a1_sym = -2.159224 * (1/tau)
a2_sym = -5.015588 * (1/tau)
# 'b' is unnecessary
v_sym_pi, v_sym_pi2 = v_evolve(
    v_1_factory(a1_sym, a2_sym, None, None, False))

a1_asym = 5.263022 * (1/tau)
b1_asym = 17.850535 * (1/tau)
a2_asym = -16.809353 * (1/tau)
b2_asym = 15.634390 * (1/tau)
v_asym_pi, v_asym_pi2 = v_evolve(
    v_1_factory(a1_asym, a2_asym, b1_asym, b2_asym, True))

# First and Second order; symmetric
a1_sym2 = 10.804433
b1_sym2 = 6.831344
c1_sym2 = 2.174538
a2_sym2 = 10.925826
b2_sym2 = 6.806775
c2_sym2 = -0.02696178
def v_sym2_minifactory(theta, a, b, c):
    def v_sym2(t):
        _1 = theta / 2
        _2 = (a - _1)   * cos((2 * pi  / tau) * t)
        _3 = (b - a)    * cos((4 * pi  / tau) * t)
        _4 = (c - b)    * cos((6 * pi  / tau) * t)
        _5 = -c         * cos((8 * pi  / tau) * t)
        return _1 + _2 + _3 + _4 + _5

    return v_sym2

v_sym2_pi_f = v_sym2_minifactory(theta1, a1_sym2, b1_sym2, c1_sym2)
v_sym2_pi2_f = v_sym2_minifactory(theta2, a2_sym2, b2_sym2, c2_sym2)

v_sym2_pi, v_sym2_pi2 = v_evolve([v_sym2_pi_f, v_sym2_pi2_f])

## Plotting
# zeros = []
# for t in ts:
#     zeros.append(0)

def decorate(p, title):
    zeros = []
    for t in ts:
        zeros.append(0)
    p.plot(ts, zeros, 'k-', lw=2)   # zero line
    p.legend(loc='best')
    p.xlabel('time')
    p.ylabel('v(t)')
    p.title(title)


f1 = p.figure(1)
p.plot(ts, v_sym_pi, 'r-', label='v_pi(t)')
p.plot(ts, v_sym_pi2, 'r--', label='v_pi/2(t)')
decorate(p,'Symmetric driving force, first order; theta = pi, pi/2')

f2 = p.figure(2)
p.plot(ts, v_asym_pi, 'b-', label='v_pi(t)')
p.plot(ts, v_asym_pi2, 'b--', label='v_pi/2(t)')
decorate(p,'Asymmetric driving force, first order; theta = pi, pi/2')

f3 = p.figure(3)
p.plot(ts, v_sym2_pi, 'g-', label='v_pi(t)')
p.plot(ts, v_sym2_pi2, 'g--', label='v_pi/2(t)')
decorate(p,'Symmetric driving force, first and second order; theta = pi, pi/2')
p.axis([0.0,1.0, -20.0,20.0])

base = "qc2-fig"
ext = ".png"
decors = [5, 6, 7]
names = mapBothPend(base, decors, ext)

figs = [f1, f2, f3]
promptSaveFigs(names, figs)

p.show()
