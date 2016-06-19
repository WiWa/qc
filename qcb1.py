from numpy import *
from scipy import integrate
import pylab as p

# norm squared
def nsq(x):
    return dot(conjugate(x), x)
def mnsq(lst):
    return map(nsq, lst)

# Params in THz
w = 10e12   # omega
W = 1e12    # Omega
D = 1e12    # Delta
# others
l = sqrt(W**2 + (D - w)**2) # lambda ~ 9.05e12 Hz
d = sqrt(D**2 + W**2)       # delta  ~ 1.41e12
N = W**2 + (d - D)**2 # ~ 1.08e12 ** 2, I'm assuming N^2 was a typo

i_ = 1.0j
def iw(t):
    return i_*w*t

def _X_plus(t):
    c = 1.0/sqrt(N)
    v = array([ W,
                (d - D)*exp(iw(t)) ])
    return c*v
def _X_minus(t):
    c = 1.0/sqrt(N)
    v = array([ (d - D)*exp(-iw(t)),
                -W ])
    return c*v

# the coeffs of X+/- in psi(t)
def psi_a(t):
    x = cos(l*t/2.) - (i_/l)*(d - (w*D)/d)*sin(l*t/2.)
    return x * exp(-iw(t)/2.)
def psi_b(t):
    x = ((i_*w*W)/(l*d))*sin(l*t/2.)
    return x * exp( iw(t)/2.)

def psi(t):
    a = psi_a(t) * _X_plus(t)
    b = psi_b(t) * _X_minus(t)
    return a + b


# exact solutions
def a_e(t):
    a_ = W / sqrt(N)
    b_ = cos(l * t/2.) - (i_/l)*(d - w)*sin(l * t/2.)
    c_ = exp(-iw(t)/2.)
    return a_*b_*c_

def b_e(t):
    a_ = (d - D) / sqrt(N)
    b_ = cos(l * t/2.) - (i_/l)*(d + w)*sin(l * t/2.)
    c_ = exp(iw(t)/2.)
    return a_*b_*c_

# X, not chi.
# X wraps the coupled diff eqs.
def dX_dt(t, X):
    # Actual "a(t)", "b(t)". c is constant.
    a = X[0]
    b = X[1]
    c = -1.0j/2

    a_dot = D*a + W*exp(-1.0j*w*t)*b
    b_dot = W*exp(1.0j*w*t)*a - D*b
    return c*array([ a_dot, b_dot ])

# X-nought starts off at a_0, b_0
X_0 = array([ a_e(0.), b_e(0.)])

# Integration "params"
steps = 100.
t0 = 0
t1 = 2.e-12     # 2 ps
dt = t1/steps     # 100 steps

print(X_0)

# Init integrator
sol = integrate.ode(dX_dt).set_integrator('zvode')
sol.set_initial_value(X_0, t0)

# Intermediate variables
X_ = []
a_e_ = []
b_e_ = []
psi_ = []

# Evolve X, stashing info in X_.
# As we evolve X through t, obtain a_e, b_e, and psi info.
while sol.successful() and sol.t < t1:
    t_ = sol.t + dt
    res = sol.integrate(t_)

    X_.append(res)
    a_e_.append(a_e(t_))
    b_e_.append(b_e(t_))
    psi_.append(psi(t_))

X = array(X_)
ts = linspace(t0, t1, len(X))          # time as x coordinate

a, b = X.T                        # Transverse
psi0 = [x[0] for x in psi_]
psi1 = [x[1] for x in psi_]

print(sol.successful())

## Plotting

f1 = p.figure()

p.plot(ts, mnsq(a), 'rx', label='a(t)^2')
p.plot(ts, mnsq(b), 'bx', label='b(t)^2')

p.plot(ts, mnsq(psi0), 'm-', label='psi(t)[0]^2')
p.plot(ts, mnsq(psi1), 'c-', label='psi(t)[1]^2')

p.legend(loc='best')
p.xlabel('time (ps)')
p.ylabel('value')
p.title('[a(t), b(t)] = psi(t); components squared')

print("Exit plot to get option to save plot.")

p.show()

saveFig = raw_input("Enter 'y' to save figure. Discarded otherwise: ")
if saveFig == "y":
    print("Saved as 'qcb1-run.png'.")
    f1.savefig('qcb1-run.png')
else:
    print("Discarded.")
