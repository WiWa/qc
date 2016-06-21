import os, sys, time
from numpy import *
from scipy import integrate
from scipy.constants import hbar, pi
import matplotlib.pyplot as p
from matplotlib.widgets import Slider, Button

# sys.path.append("/home/arbiter/qc")
# from getch import getch
# PARAMS
num_thetas = 2
num_deltas = 50

theta_min = 0.
theta_max = 2*pi
Delta_min = 0.1e-35
Delta_max = 5e-33

# electron relaxation time between singlet triplet state
# s. pasini paper referenced t in units of "tau_p"??
tau = 1.

# Delta is arbitrary for now
# D = 1e12    # Delta
D = 1e-33

i_ = 1.0j

delta_str = "Delta = " + str(D)
sym_title = "Chi Symmetric; Real values; " + delta_str
asym_title = "Chi Antisymmetric; Real values; " + delta_str

var_txt = """
Delta = {D}
""".format(**locals())

def iw(t):
    return i_*w*t

def normalize(v):
    magnitude = norm(v)
    return map(lambda v_i: float(v_i)/magnitude, v)

def sq(x):
    return x**2

def csq(x):
    return dot(x, conjugate(x))

def norm(x):
    return sqrt(csq(x))

def decorate(p, title, xlabel="time", ylabel="values"):
    p.legend(loc='best', title=var_txt)
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



# def Chi_plus_dot_f(plus, minus, Delta, Xt):
#     return Delta*plus + (hbar / tau)*minus*Xt
#
# def Chi_minus_dot_f(plus, minus, Delta, Xt):
#     return (hbar / tau)*plus*Xt - Delta*minus

def dChis_dt_factory(X, Delta):
    # wraps the coupled diff eqs.
    def dChis_dt(t, Chi):
        plus = Chi[0]      # Chi_plus component
        minus = Chi[1]      # Chi_minus component
        c = -1.0j/hbar

        def Chi_plus_dot_f(plus, minus):
            return Delta*plus + (hbar / tau)*minus*X(t)

        def Chi_minus_dot_f(plus, minus):
            return (hbar / tau)*plus*X(t) - Delta*minus

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
dChiSym_dt = dChis_dt_factory(X_s_f, D)

# X_sb_f = X_factory(theta1, a2_sym, None, False)
# dChiSym2_dt = dChis_dt_factory(X_s_f)

# Antisymmetric
X_a_f = X_factory(theta1, a1_asym, b1_asym, True)
dChiAsym_dt = dChis_dt_factory(X_a_f, D)


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

memoize = False

# print("press 'y' to memoize")
# doMemo = getch()
# if doMemo == 'y':
#     memoize = True
#     print("Memoizing")


### Work with Antisymmetric pulse

# Chi, suc = intAndPlotVec2(dChiAsym_dt, Chi_0, t0, t1, p, asym_title)


### Fidelity vs Theta


thetas = linspace(theta_min, theta_max, num_thetas)
Deltas = linspace(Delta_min, Delta_max, num_deltas)


## Data to take
## Keep track of these guys throughout the app ;)
Chi_ps = []
Chi_ms = []
max_fids = []
avg_fids = []
fid_lists = []

if memoize:
    start = time.time()

    for i in range(len(Deltas)):

        length = str(len(Deltas))
        if i % 10 is len(Deltas) / 10:
            print("... " + str(i) + "/" + length)

        Chi_ps.append([])
        Chi_ms.append([])
        max_fids.append([])
        avg_fids.append([])
        fid_lists.append([])

        for j in range(len(thetas)):
            Delta = Deltas[i]
            theta = thetas[j]

            X_a_f = X_factory(theta, a1_asym, b1_asym, True)
            dChiAsym_dt = dChis_dt_factory(X_a_f, Delta)
            ChiA_, suc = wrapIntegrate(dChiAsym_dt, Chi_0, t0, t1)
            ChiA_p, ChiA_m = array(ChiA_).T

            fidelity = map(norm, ChiA_p)

            Chi_ps[i].append(ChiA_p)
            Chi_ms[i].append(ChiA_m)
            max_fids[i].append(max(fidelity))
            avg_fids[i].append(average(fidelity))
            fid_lists[i].append(fidelity)

    print("Time taken to memoize: " + str( time.time() - start ))

def makeLine(c, xs):
    return [c for x in xs]

delta_txt = """Delta ranges from
    {Delta_min} to {Delta_max}""".format(**locals())


# s = fid_lists[i0][j0]
# l, = p.plot(ts, s, lw=2, color='red')

def annotateSumProb(t, sum_prob):
    val = sum_prob[len(sum_prob)/2]
    s = "Sum of probabilities: " + str(val)
    t.set_text(s)

if memoize:
    i0 = num_deltas / 2
    j0 = num_thetas / 2

    Delta0 = Deltas[i0]
    theta0 = thetas[j0]

    Chi_ps_0 = Chi_ps[i0][j0]
    Chi_ms_0 = Chi_ms[i0][j0]
    max_fid_0 = makeLine(max_fids[i0][j0], ts)
    avg_fid_0 = makeLine(avg_fids[i0][j0], ts)
    fid_0 = fid_lists[i0][j0]
else:
    Delta0 = (Delta_max - Delta_min) / 2
    theta0 = (theta_max - theta_min) / 2

    X_a_f = X_factory(theta0, a1_asym, b1_asym, True)
    dChiAsym_dt = dChis_dt_factory(X_a_f, Delta0)
    ChiA_, suc = wrapIntegrate(dChiAsym_dt, Chi_0, t0, t1)
    ChiA_p, ChiA_m = array(ChiA_).T
    fidelity = map(norm, ChiA_p)
    fidelity_down = map(norm, ChiA_m)

    Chi_ps_0 = map(csq, ChiA_p)
    Chi_ms_0 = map(csq, ChiA_m)
    sum_prob_0 = makeLine(average(Chi_ps_0) + average(Chi_ms_0), ts)
    max_fid_0 = makeLine(max(fidelity), ts)
    avg_fid_0 = makeLine(average(fidelity), ts)
    fid_0 = fidelity
    fid_d_0 = fidelity_down

fig, axarr = p.subplots(3, sharex=True, figsize=(12, 10))
p.subplots_adjust(left=0.25, bottom=0.30)

v_Chi_p, = axarr[0].plot(ts, Chi_ps_0, 'r-', label="+ Probablity")
v_Chi_m, = axarr[0].plot(ts, Chi_ms_0, 'b-', label="- Probablity")
v_sum_prob, = axarr[0].plot(ts, sum_prob_0, 'g--', label="Sum")
axarr[0].axis([0., 1., -0.1, 1.2])
z = sum_prob_0[len(sum_prob_0)/2]
sum_prob_text = axarr[0].text(0.3, 1.1, "Sum of probabilities: " + str(z), ha='center', va='center')
axarr[0].set_title("Chi Components Squared (Probabilities)")
l = axarr[0].legend(loc='upper right', fancybox=True)
l.draggable(True)
l.get_frame().set_alpha(0.5)

v_max_fid, = axarr[1].plot(ts, max_fid_0, "b--", label="Max Fidelity")
v_avg_fid, = axarr[1].plot(ts, avg_fid_0, "g--", label="Average Fidelity")
v_fid, = axarr[1].plot(ts, fid_0, lw=2, color='red', label="Fidelity (|Chi+>)")
axarr[1].axis([0., 1., -0.1, 1.2])
axarr[1].set_title("Fidelity (|Chi+>) vs Time")
axarr[1].set_ylabel("|Chi+>")
l = axarr[1].legend(loc='lower left', fancybox=True)
l.draggable(True)
l.get_frame().set_alpha(0.5)

v_fid_d, = axarr[2].plot(ts, fid_d_0, lw=2, color='red', label="|Chi->")
decorate(p, "|Chi-> vs Time", xlabel="Time (tau)")
axarr[2].axis([0., 1., -0.1, 1.2])
axarr[2].set_ylabel("|Chi->")
l = axarr[2].legend(loc='lower left', fancybox=True)
l.draggable(True)
l.get_frame().set_alpha(0.5)




axcolor = 'lightgoldenrodyellow'
axDelta = p.axes([0.25, 0.10, 0.65, 0.03], axisbg=axcolor)
axtheta = p.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

d_cor = 1e34

sDelta = Slider(axDelta, "Delta ({})".format(1./d_cor), Delta_min*d_cor, Delta_max*d_cor,
                valinit=Delta0*d_cor)
stheta = Slider(axtheta, "theta", theta_min, theta_max,
                valinit=theta0)

def update(val):

    # Memoization!
    if memoize:
        i = rangeAdapt(sDelta.val / d_cor, Delta_min, Delta_max, Deltas)
        j = rangeAdapt(stheta.val, theta_min, theta_max, thetas)
        Delta = Deltas[i]
        theta = thetas[j]

        v_Chi_p.set_ydata(Chi_ps[i][j])
        v_Chi_m.set_ydata(Chi_ms[i][j])
        v_max_fid.set_ydata(max_fids[i][j])
        v_avg_fid.set_ydata(avg_fids[i][j])
        v_fid.set_ydata(fid_lists[i][j])

    # Dynamic?!
    else:
        Delta = sDelta.val / d_cor
        theta = stheta.val

        X_a_f = X_factory(theta, a1_asym, b1_asym, True)
        dChiAsym_dt = dChis_dt_factory(X_a_f, Delta)
        ChiA_, suc = wrapIntegrate(dChiAsym_dt, Chi_0, t0, t1)
        ChiA_p, ChiA_m = array(ChiA_).T
        fidelity = map(norm, ChiA_p)
        fidelity_down = map(norm, ChiA_m)

        Chi_ps = map(csq, ChiA_p)
        Chi_ms = map(csq, ChiA_m)
        sum_prob = makeLine(average(Chi_ps) + average(Chi_ms), ts)
        v_Chi_p.set_ydata(Chi_ps)
        v_Chi_m.set_ydata(Chi_ms)
        v_sum_prob.set_ydata(sum_prob)
        v_max_fid.set_ydata(max(fidelity))
        v_avg_fid.set_ydata(average(fidelity))
        v_fid.set_ydata(fidelity)
        v_fid_d.set_ydata(fidelity_down)
        annotateSumProb(sum_prob_text, sum_prob)

    fig.canvas.draw_idle()
sDelta.on_changed(update)
stheta.on_changed(update)

### BUTTONS
def addParens(s):
    return "(%s)" % (s)
def getNextName(filepath, ext):
    uniq = 0
    filepath_mod = filepath + "." + ext
    while os.path.exists(filepath_mod):
        uniq += 1
        filepath_mod = "%s %s.%s" % (filepath, addParens(uniq), ext)
    return filepath_mod

def save(event):
    filename = "../figures/fidelity-fig"
    ext = 'png'
    filename = getNextName(filename, ext)
    fig.savefig(filename)
    print("Saved as: ")
    print(filename)

def reset(event):
    sDelta.reset()
    stheta.reset()

save_ax = p.axes([0.8, 0.025, 0.1, 0.04])
save_btn = Button(save_ax, 'Save', color=axcolor, hovercolor='0.975')
save_btn.on_clicked(save)

reset_ax = p.axes([0.6, 0.025, 0.1, 0.04])
reset_btn = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.975')
reset_btn.on_clicked(reset)


p.show()
