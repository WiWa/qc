from time import time
import numpy as np

l = [1 for i in range(10000)]

a = np.array(l)
b = np.array(l)

start = time()
[np.dot(a, b) for i in range(1000)]
print(time() - start)

start = time()
[a * b for i in range(1000)]
print(time() - start)


start = time()
[np.dot(a, b) for i in range(1000)]
print(time() - start)
