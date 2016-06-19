from numpy import *
from scipy import integrate
import pylab as p

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

def X_plus(t):
    c = 1.0/sqrt(N)
    v = array([ W,
                (d - D)*exp(iw(t)) ])
    return c*v
def X_minus(t):
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
    a = psi_a(t) * X_plus(t)
    b = psi_b(t) * X_minus(t)
    return a + b

# exact solutions
def ae(t):
    a_ = W / sqrt(N)
    b_ = cos(l * t/2.) - (i_/l)*(d - w)*sin(l * t/2.)
    c_ = exp(-iw(t)/2.)
    return a_*b_*c_

def be(t):
    a_ = (d - D) / sqrt(N)
    b_ = cos(l * t/2.) - (i_/l)*(d + w)*sin(l * t/2.)
    c_ = exp(iw(t)/2.)
    return a_*b_*c_

def tx_prob(t):
    a_ = (w * W) / (l * d)
    b_ = sin(l * t / 2.)
    return a_**2 * b_**2

# def dX_dt(X, t=0):
def dX_dt(t, X):
    a = X[0]
    b = X[1]
    c = -1.0j/2

    # problem is a and b
    a_dot = D*a + W*exp(-1.0j*w*t)*b
    b_dot = W*exp(1.0j*w*t)*a - D*b
    return c*array([ a_dot, b_dot ])


#
# t = linspace(0, 20,  1000)          # time
# X_0 = array([ 1., 0.])
X_0 = array([ ae(0.), be(0.)])

# X, infodict = odeintz(dX_dt, X_0, t, full_output=True)
# print(infodict['message'])              # >>> 'Integration successful.'

steps = 100.
t0 = 0
t1 = 2.e-12     # 2 ps
dt = t1/steps     # 100 steps

sol = integrate.ode(dX_dt).set_integrator('zvode')
sol.set_initial_value(X_0, t0)
a2 = []
b2 = []
a_e = []
b_e = []
X_ = []
pa = []
pb = []
psis = []
txs = []
while sol.successful() and sol.t < t1:
    t_ = sol.t + dt
    res = sol.integrate(t_)
    X_.append(res)
    a2.append(res[0] * X_plus(t_))
    b2.append(res[1] * X_minus(t_))
    a_e.append(ae(t_))
    b_e.append(be(t_))
    pa.append(psi_a(t_))
    pb.append(psi_b(t_))
    psis.append(psi(t_))
    txs.append(tx_prob(t_))

X = array(X_)
ts = linspace(t0, t1, len(X))          # time
## Plotting
a, b = X.T                        # Transverse

# Actually useless lol
a2b2 = []
for i in range(len(a2)):
    a2b2.append(a2[i] + b2[i])
# a_b = []
# for i in range(len(a)):
#     a_b.append(a[i] + b[i])

# norm squared
def nsq(x):
    return dot(conjugate(x), x)
def mnsq(lst):
    return map(nsq, lst)

print(sol.successful())
print(nsq(X_plus(0)))
print(tx_prob( pi / l ))
# print(X);
psi0 = [x[0] for x in psis]
psi1 = [x[1] for x in psis]

f1 = p.figure()                             # pylab does matplotlib I guess
p.plot(ts, mnsq(a), 'rx', label='a(t)^2')
p.plot(ts, mnsq(b), 'bx', label='b(t)^2')
# p.plot(ts, a_e, 'g-', label='a(t) exact')
# p.plot(ts, b_e, 'y-', label='b(t) exact')
# p.plot(ts, mnsq(a), 'r-', label='a(t)^2')
# p.plot(ts, mnsq(b), 'y-', label='b(t)^2')
# p.plot(ts, txs, 'g-', label='txs')
# p.plot(ts, mnsq(psis), 'm-', label='psi(t)^2')
p.plot(ts, mnsq(psi0), 'm-', label='psi(t)[0]^2')
p.plot(ts, mnsq(psi1), 'c-', label='psi(t)[1]^2')
# p.plot(ts, pa, 'g-', label='psi_a(t)')
# p.plot(ts, pb, 'c-', label='psi_b(t)')
# p.plot(ts, mnsq(pa), 'g-', label='psi_a(t)^2')    # gives transition probs?? fits nicely
# p.plot(ts, mnsq(pb), 'g-', label='psi_b(t)^2')
# p.plot(ts, a2b2, 'yx', label='aX+ + bX-')
# p.plot(ts, mnsq(a2), 'y-', label='aX+')
# p.plot(ts, mnsq(b2), 'y-', label='bX-')
# p.grid()
p.legend(loc='best')
p.xlabel('time (ps)')
p.ylabel('value')
p.title('[a(t), b(t)] = psi(t); components squared')
# f1.savefig('a_b_components_of_psi_squared.png')
p.show()
