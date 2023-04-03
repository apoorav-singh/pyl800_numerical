## Code By: Apoorav Singh Deo
## En. No.: 2021PHZ8046

import numpy as np
from matplotlib import pyplot as plt
from math import *

import time

t1 = time.time()

h1 = 1e-4

tol = pow(10, -6)  # Tolerance limit for approximating zero is defined

# Solving for RHS First

# Function definition for simpson's (1/3) rule


def simp_1_3(b, a, h, func):

    integr = 0

    n = floor((b - a) / h)

    n = abs(n)

    print(n)

    i = 0

    h1 = (b - a) / n

    x = np.empty(n)

    x[0] = a

    while i < n:

        x[i] = i * h1 + a

        if i % 2 == 0 and i != (n - 1) and i != 0:

            integr += (h1 / 3) * 2 * func(x[i])

        elif i % 2 != 0 and i != (n - 1) and i != 0:

            integr += (h1 / 3) * 4 * func(x[i])

        else:

            integr += (h1 / 3) * func(x[i])

        i += 1
        # print(integr)
    return integr


# Case 1: (0,0,0) ----> (0,0,1)

# As function becomes zero therefore answer for this integration is 0.

c_res_1 = 0

# Case 2: (0,0,1) ----> (0,1,1)


def func_2(x):

    func = 3 * x ** 2
    return func


c_res_2 = simp_1_3(1, 0, h1, func_2)

# Case 3: (0,1,1) ----> (0,1,0)


def func_3(x):

    func = 4 * x ** 2
    return func


c_res_3 = simp_1_3(0, 1, h1, func_3)


# Case 4: (0,1,0) ----> (0,0,0)

# Due to symmetry of the problem the function is same as func_2(x) defined earlier

c_res_4 = simp_1_3(0, 1, h1, func_2)


rhs_res = c_res_1 + c_res_2 + c_res_3 + c_res_4

print("Value of the Integral in RHS {}".format((abs(rhs_res))))

t2 = time.time()

print("Time taken to execute the code {} s".format(t2 - t1))
