import sys
from numpy import *
from scipy import integrate
from scipy.constants import hbar, pi
import matplotlib.pyplot as p
from matplotlib.widgets import Slider, Button

# electron relaxation time between singlet triplet state
# s. pasini paper referenced t in units of "tau_p"??
tau = 1.

# Delta is arbitrary for now
# D = 1e12    # Delta
D = 6e-33

i_ = 1.0j

delta_str = "Delta = " + str(D)
sym_title = "Chi Symmetric; Real values; " + delta_str
asym_title = "Chi Antisymmetric; Real values; " + delta_str

def iw(t):
    return i_*w*t

def normalize(v):
    magnitude = norm(v)
    return map(lambda v_i: float(v_i)/magnitude, v)

def sq(x):
    return x**2

def norm(x):
    return sqrt(dot(x, conjugate(x)))

def decorate(p, title, xlabel="time", ylabel="values"):
    p.legend(loc='best')
    p.xlabel(xlabel)
    p.ylabel(ylabel)
    p.title(title)

def rangeAdapt(v, v_min, v_max, list):
    partial = (v - v_min) / (v_max - v_min)
    max_index = len(list) - 1
    almost = int(floor( partial * (max_index) ))
    return max(0, min(almost, max_index))

# Makes X(t) driving pulse function
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
        plus = Chi[0]      # Chi_plus component
        minus = Chi[1]      # Chi_minus component
        c = -1.0j/hbar

        def Chi_plus_dot_f(plus, minus):
            return D*plus + (hbar / tau)*minus*X(t)

        def Chi_minus_dot_f(plus, minus):
            return (hbar / tau)*plus*X(t) - D*minus

        plus_dot = Chi_plus_dot_f(plus, minus)
        minus_dot = Chi_minus_dot_f(plus, minus)

        return c*array([
                    plus_dot,
                    minus_dot,
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

# X_sb_f = X_factory(theta1, a2_sym, None, False)
# dChiSym2_dt = dChis_dt_factory(X_s_f)

# Antisymmetric
X_a_f = X_factory(theta1, a1_asym, b1_asym, True)
dChiAsym_dt = dChis_dt_factory(X_a_f)

# X_a1b_f = X_factory(theta2, a1_asym, b1_asym, True)
# dChiAsymb_dt = dChis_dt_factory(X_a1b_f)
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

# Integration "params"
steps = 800.
t0 = 0.
t1 = tau
dt = t1/steps     # 100 steps

ts = linspace(t0, t1, steps + 1)

def wrapIntegrate(func, init, t0, t1):
    vals = []
    sol = integrate.ode(func).set_integrator('zvode')
    sol.set_initial_value(init, t0)
    while sol.successful() and sol.t < t1:
        t_ = sol.t + dt
        res = sol.integrate(t_)
        vals.append(res)
    return (vals, sol.successful())

def plotVec2(v0, v1, p, title, label0="|0>", label1="|1>"):
    p.figure()
    p.plot(ts, v0, 'r-', label=label0)
    p.plot(ts, v1, 'b-', label=label1)
    decorate(p, title)

def intAndPlotVec2(func, init, t0, t1, p, title,
                    label0="|0>", label1="|1>"):
    F_, success = wrapIntegrate(func, init, t0, t1)
    F0, F1 = array(F_).T
    plotVec2(F0, F1, p, title, label0, label1)

    return (F_, success)


# Chi_p, Chi_m = array(Chi).T
#
# fidelity = map(norm, Chi_p)

# p.figure()
# p.plot(ts, fidelity, 'r-', label="Fidelity")
# decorate(p, "Fidelity; Antisym, theta = pi")

## Plotting

# f1 = p.figure(1)
# p.plot(ts, ChiS_p, 'r-', label='Chi_+')
# p.plot(ts, ChiS_m, 'b-', label='Chi_-')
# decorate(p, sym_title)
#
# f2 = p.figure(2)
# fidS = map(norm, ChiS_p)
# p.plot(ts, fidS, 'r-', label='Fidelity = |<Chi|0>')
# decorate(p, sym_title)
# print("Max Fidelity Sym: " + str(max(fidS)))
#
# f1 = p.figure(3)
# p.plot(ts, ChiA_p, 'r-', label='Chi_+')
# p.plot(ts, ChiA_m, 'b-', label='Chi_-')
# decorate(p, asym_title)
#
# f2 = p.figure(4)
# fidA = map(norm, ChiA_p)
# p.plot(ts, fidA, 'r-', label='Fidelity = |<Chi|0>')
# decorate(p, asym_title)
# print("Max Fidelity Antisym: " + str(max(fidA)))

Chi, suc = intAndPlotVec2(dChiAsym_dt, Chi_0, t0, t1, p, asym_title)

### Fidelity vs Theta

do_s = False
do_a = True

thetas = linspace(0., pi, 42)
max_fidsS = []
avg_fidsS = []
max_fidsA = []
avg_fidsA = []

fid_lists = []

for theta in thetas:
    #Sym
    if do_s:
        X_s_f = X_factory(theta, a1_asym, None, False)
        dChiSym_dt = dChis_dt_factory(X_s_f)
        ChiA_, success = wrapIntegrate(dChiSym_dt, Chi_0, t0, t1)
        ChiS_p, ChiS_m = ChiS.T
        fidsS = map(norm, ChiS_p)
        max_fidsS.append(max(fidsS))
        avg_fidsS.append(average(fidsS))
    if do_a:
    #Asym
        X_a_f = X_factory(theta, a1_asym, b1_asym, True)
        dChiAsym_dt = dChis_dt_factory(X_a_f)
        ChiA_, success = wrapIntegrate(dChiAsym_dt, Chi_0, t0, t1)
        ChiA_p, ChiA_m = array(ChiA_).T
        fidsA = map(norm, ChiA_p)
        max_fidsA.append(max(fidsA))
        avg_fidsA.append(average(fidsA))

        fid_lists.append(fidsA)

if do_a or do_s:
    p.figure()
if do_s:
    p.plot(thetas, max_fidsS, 'r-', label='Max Fidelity S')
    p.plot(thetas, avg_fidsS, 'r--', label='Average Fidelity S')
if do_a:
    p.plot(thetas, max_fidsA, 'b-', label='Max Fidelity A')
    p.plot(thetas, avg_fidsA, 'b--', label='Average Fidelity A')
decorate(p, "Max/Avg fidelity vs theta", xlabel="Theta", ylabel="Fidelity")


fig, ax = p.subplots()
p.subplots_adjust(left=0.25, bottom=0.35)


i = 21
theta0 = thetas[i]
s = fid_lists[i]
l, = p.plot(ts, s, lw=2, color='red')
decorate(p, "Fidelity vs Time", xlabel="Time (tau)", ylabel="Fidelity")


axcolor = 'lightgoldenrodyellow'
axtheta = p.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

stheta = Slider(axtheta, "Theta", 0, pi, valinit=thetas[i])

def update(val):
    i = rangeAdapt(stheta.val, 0, pi, thetas)
    theta = thetas[i]
    l.set_ydata(fid_lists[i])
    fig.canvas.draw_idle()
stheta.on_changed(update)

p.show()
