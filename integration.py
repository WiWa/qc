
from mod import *
import sympy as sp
from scipy import integrate as scint



x = sp.Symbol('x')

def f(x):
    """
    Cheeky dynamic binding allows f(x) to take both numbers and symbols.
    """
    return sp.exp(-x) * sp.sin(3 * x)

f_i = sp.integrate(f(x), x)

print(f_i)

r = f_i.evalf(subs={x:2*sp.pi}) - f_i.evalf(subs={x:0})

print("{}, {}".format(r, type(r)))

r2, r2_err = scint.quad(f, 0, 2*sp.pi)

print("{}, {}".format(r2, type(r2)))

print("Diff between sym and num: {}".format(r - r2))
