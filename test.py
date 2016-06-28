
import numpy as np
from scipy.linalg import expm
from time import time

def trotter(Ms, n):
    n_ = np.array(1./n)
    Ms_ = np.array([np.dot(n_, M) for M in Ms])
    # print(Ms_[1])
    eMs = map(expm, Ms_)
    prod = reduce(np.dot, eMs)
    return np.linalg.matrix_power(prod, n)

def trotter2(Ms, n):
    n_ = np.array(1./n)
    eMs = map(expm, [np.dot(n_, M) for M in Ms])
    prod = reduce(np.dot, eMs)
    return np.linalg.matrix_power(prod, n)

def BCHtest(Ms, order=5):
    C = Ms[0]
    for i in range(1, len(Ms)):
        C = BCH(C, Ms[i], order=order)
    return expm(C)

k1 = np.array(1/2.,np.complex128)
k2 = np.array(1/12.,np.complex128)
k3 = np.array(-1/12.,np.complex128)
k4 = np.array(-1/24.,np.complex128)
k5 = np.array(-1/720.,np.complex128)
k6 = np.array(1/360.,np.complex128)
k7 = np.array(1/120.,np.complex128)
# Baker-Campbell-Hausdorff approx
def BCH(A, B, order=4):
    # start = time.time()

    c1 = np.dot(k1,comm(A, B))
    c2 = np.dot(k2,comm(A, c1))
    c3 = np.dot(k3,comm(B, c1))
    res = A + B + c1 + c2 + c3
    if order > 3:
        c4 = np.dot(k4,comm(B, c2))
        res += c4
    if order > 4:
        c5 = np.dot(k5,(bch5(A, B) + bch5(B, A)))
        res += c5
    if order > 5:
        c6 = np.dot(k6,(bch6(A, B) + bch6(B, A)))
        res += c6
    if order > 6:
        c7 = np.dot(k7,(bch7(A, B) + bch7(B, A)))
        res += c7
    # if profiling:
    #     bch_time[0] += time.time() - start
    return res

def bch5(A, B):
    return comm(B,comm(B,comm(B,comm(B,A))))
def bch6(A, B):
    return comm(A,comm(B,comm(B,comm(B,A))))
def bch7(A, B):
    return comm(B,comm(A,comm(B,comm(A,B))))

# Commutator
def comm(A, B):
    return np.dot(A, B) - np.dot(B, A)

def genM():
    a = np.random.random()
    b = np.random.random()
    return np.array([[a,b],[b,-a]])

np.random.random()
N = 10000
Ms = [genM() for i in range(N)]
n = 100000

# s = time()
# r = trotter(Ms, n)
# print(time() - s)
#
# s = time()
# r = trotter2(Ms, n)
# print(time() - s)
#
# s = time()
# r = trotter(Ms, n)
# print(time() - s)

cor = 10000.
Ms = map(lambda m: np.dot(1./cor, m), Ms)
s = time()
r = BCHtest(Ms, order=4)
print(time() - s)
s = time()
r2 = BCHtest(Ms, order=5)
print(time() - s)
print(np.true_divide(r - r2, r))
