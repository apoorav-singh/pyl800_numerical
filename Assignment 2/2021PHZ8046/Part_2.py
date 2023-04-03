## Code By: Apoorav Singh Deo
## En. No.: 2021PHZ8046

import numpy as np
from matplotlib import pyplot as plt
from math import *

import time

t1 = time.time()

tol = pow(10, -6)  # Tolerance limit for approximating zero is defined
arry_lim = 1000000  # Safe Array Size
h = 1e-3  # [Step Size]

# In this block of the code, infinity would be materialized

g = 0.00609  # [mi/s^2]
R = 3960  # [miles]

z = np.zeros(arry_lim)
pl_f_z = np.zeros(arry_lim)
trap = 0


def func_v_2(z):
    f_z = 2 * g * R * (1 / pow(z, 2))
    return f_z


z[0] = 1
i = 0

while pl_f_z[i] < tol:

    pl_f_z[i] = func_v_2(z[i])

    if z[i] >= 695:
        break

    z[i + 1] = z[i] + h
    # z_i+1 is placed here to avoid mismatch of array size

    i += 1

# Implementing Trapezoidal Method

i = 1

while True:
    trap += (pl_f_z[i - 1] + pl_f_z[i]) * h / 2

    if z[i] >= 695:
        break

    i += 1

print("Escape velocity  = {} miles/s".format(sqrt(trap)))

plt.plot(z, pl_f_z, label="Integrand")
plt.xlabel(r"$z\ \rightarrow$")
plt.ylabel(r"$f(z)\ \rightarrow$")
plt.grid(1)
plt.title(r"$z \approx 694.5\ gives\ us\ tolerance\ \approx 9 \times 10^{-5}$")
plt.legend()

t2 = time.time()

print("Time taken to execute the code {} s".format(t2 - t1))
