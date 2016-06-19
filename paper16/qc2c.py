from numpy import *
from scipy import integrate
from scipy.constants import hbar, pi
import pylab as p

# electron relaxation time between singlet triplet state
# s. pasini paper referenced t in units of "tau_p"??
tau = 1e-11
tau=1

# Params in THz
w = 1e11
D = 1e12    # Delta from long ago
D = 1e-40

i_ = 1.0j
def iw(t):
    return i_*w*t


def X_factory(theta_f, a, b, antisym):
    def X_sym(t):
        _1 = theta_f(t) / 2
        _2 = (a - _1) * cos((2 * pi  / tau) * t)
        _3 = -a * cos((4 * pi  / tau) * t)
        return _1 + _2 + _3

    return X_sym

def dChis_dt_factory(X):
    # wraps the coupled diff eqs.
    def dChis_dt(t, Chi):
        a = Chi[0]      # Chi_plus  dot
        b = Chi[1]      # Chi_minus dot
        c = -1.0j/hbar  # 1/j = -j

        a_dot = D*a + (hbar / tau)*b*X(t)
        b_dot = (hbar / tau)*a*X(t) - D*b
        return c*array([ a_dot, b_dot ])

    return dChis_dt

def theta_f(t):
    return w * t

a1_sym = -2.159224 * (1/tau)
 # ZVODE--  At T (=R1) and step size H (=R2), the
 #       corrector convergence failed repeatedly
 #       or with abs(H) = HMIN
# a1_sym = 0
# a2_sym = -5.015588 * (1/tau) # for pi/2

# (First-order) Symmetric
X1_sym = X_factory(theta_f, a1_sym, None, False)

dChiSym_dt = dChis_dt_factory(X1_sym)

# Setup orthogonal spinors
X_pi = X1_sym(pi)
delta = sqrt(D**2 + ((hbar / tau)**2)*(X_pi**2))
N_chi = 1/sqrt( (hbar*X_pi)**2 + (delta - D)**2 )
Chi_plus_0 = N_chi * array([ hbar * X_pi, (delta - D)*tau ])
Chi_minus_0 = N_chi * array([ (delta - D)*tau, -hbar * X_pi ])
Chi_0 = Chi_plus_0 + Chi_minus_0

# Integration "params"
steps = 100.
t0 = 0
t1 = tau
dt = t1/steps     # 100 steps

# Init integrator
sol = integrate.ode(dChiSym_dt).set_integrator('zvode')
sol.set_initial_value(Chi_0, t0)

# Intermediate variables
Chi_ = []

# Evolve X, stashing info in X_.
# As we evolve X through t, obtain a_e, b_e, and psi info.
while sol.successful() and sol.t < t1:
    t_ = sol.t + dt
    res = sol.integrate(t_)

    Chi_.append(res)

success = sol.successful()
print(success)
if success:
    Chi = array(Chi_)
    ts = linspace(t0, t1, len(Chi))          # time as x coordinate

    a, b = Chi.T                        # Transverse=


    ## Plotting

    f1 = p.figure()
    p.plot(ts, a, 'r-', label='Chi[0]')
    p.plot(ts, b, 'b-', label='Chi[1]')

    p.legend(loc='best')
    p.xlabel('time ')
    p.ylabel('value')
    p.title('Chi_pi symmetric')

    p.show()
