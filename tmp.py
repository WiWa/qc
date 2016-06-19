from numpy import *
from scipy import integrate
import pylab as p

# du/dt =  a*u -   b*u*v
# dv/dt = -c*v + d*b*u*v
# u: number of preys (for example, rabbits)
# v: number of predators (for example, foxes)
# a, b, c, d are constant parameters defining the behavior of the population:
# a is the natural growing rate of rabbits, when there's no fox
# b is the natural dying rate of rabbits, due to predation
# c is the natural dying rate of fox, when there's no rabbit
# d is the factor describing how many caught rabbits let create a new fox

# params
a = 1
b = 0.1
c = 1.5
d = 0.75

def dX_dt(X, t=0):
    """ Return growth rate of fox and rabbit populations """
    u = X[0]
    v = X[1]
    return array([  a*u -   b*u*v   ,
                    -c*v +  d*b*u*v ])

# how the hell did we get this?
def d2X_dt2(X, t=0):
    """ Return the Jacobian matrix evaluated in X. """
    return array([[a -b*X[1],   -b*X[0]     ],
                  [b*d*X[1] ,   -c +b*d*X[0]] ])

# Population Equilibrium -- growth rate zero
# We magically found these points whee
X_f0 = array([     0. ,  0.])
X_f1 = array([ c/(d*b), a/b])
# This is doable because array([]) is magical
y = all(dX_dt(X_f0) == zeros(2) ) and all(dX_dt(X_f1) == zeros(2)) # => True

print(y)

A_f0 = d2X_dt2(X_f0)                    # >>> array([[ 1. , -0. ],
                                        #            [ 0. , -1.5]])
A_f1 = d2X_dt2(X_f1)                    # >>> array([[ 0.  , -2.  ],
                                        #            [ 0.75,  0.  ]])
# whose eigenvalues are +/- sqrt(c*a).j:
lambda1, lambda2 = linalg.eigvals(A_f1) # >>> (1.22474j, -1.22474j)
# They are imaginary numbers. The fox and rabbit populations are periodic as follows from further
# analysis. Their period is given by:
T_f1 = 2*pi/abs(lambda1)                # >>> 5.130199


## Integration
t = linspace(0, 25,  2000)              # time
X0 = array([10, 5])                     # initials conditions: 10 rabbits and 5 foxes
X, infodict = integrate.odeint(dX_dt, X0, t, full_output=True)
print(infodict['message'])              # >>> 'Integration successful.'
## Plotting
rabbits, foxes = X.T                        # Transverse

## Integration
X02 = array([40, 1])
X2, infodict = integrate.odeint(dX_dt, X02, t, full_output=True)
## Plotting
rabbits2, foxes2 = X2.T                        # Transverse

f1 = p.figure()                             # pylab does matplotlib I guess
p.plot(t, rabbits, 'r-', label='Rabbits')
p.plot(t, foxes  , 'b-', label='Foxes')
p.plot(t, rabbits2, 'y-', label='Rabbits0')
p.plot(t, foxes2  , 'g-', label='Foxes0')
p.grid()
p.legend(loc='best')
p.xlabel('time')
p.ylabel('population')
p.title('Evolution of fox and rabbit populations')
f1.savefig('rabbits_and_foxes_1.png')
