## Code By: Apoorav Singh Deo
## En. No.: 2021PHZ8046

import numpy as np
from matplotlib import pyplot as plt
from math import *
import time

t1 = time.time()

# Function definition


def func_xy(y, x):

    func = 4 * y ** 2

    return func


# This function is modified simpson function, in which function can take f(x,y) value. Rest all is same


def simp_1_3m(b, a, h1, y, func):

    integr = 0

    n = floor((b - a) / h1)

    n = abs(n)

    i = 0

    x = np.empty(n)

    x[0] = a

    while i < n:

        x[i] = i * h1 + a

        if i % 2 == 0 and i != (n - 1) and i != 0:

            integr += 2 * func(y, x[i])

        elif i % 2 != 0 and i != (n - 1) and i != 0:

            integr += 4 * func(y, x[i])

        else:

            integr += func(y, x[i])

        i += 1
    return integr


def d_simp_1_3(bx, ax, by, ay, hx, hy, func):
    # --------------------------------------
    # ax: Lower bound dx
    # bx: Upper bound dx
    # by: Upper bound dy
    # ay: Lower bound dy
    # hx: Step Size x-axis
    # hy: Step Size y-axis
    # func: Input function of the Integrand
    # ---------------------------------------

    # Will store the integration value
    integr_y = 0  #

    nx = floor((bx - ax) / hx)  # Steps in x axis

    nx = abs(nx)  # Taking absolute value to avoid error with indexing

    ny = floor((by - ay) / hy)  # Steps in y axis

    ny = abs(ny)  # Taking absolute value to avoid error with indexing

    i = 0  # (Inner Iteration)
    j = 0  # (Outer Iteration)

    h1x = (bx - ax) / nx  # | In this way we can also handle the Integrals
    h1y = (by - ay) / ny  # | that have b<a. As it adjusts the

    x = np.empty(nx)  #
    # Initializing the variables
    y = np.empty(ny)  #

    x[0] = ax  #
    # Initializing the variables
    y[0] = ay  #

    for j in range(ny):

        y[j] = j * h1y + ay

        if j % 2 == 0 and j != (ny - 1) and j != 0:

            integr_y += (h1x * h1y / 9) * 2 * simp_1_3m(bx, ax, h1x, y[j], func)

        elif j % 2 != 0 and j != (ny - 1) and j != 0:

            integr_y += (h1x * h1y / 9) * 4 * simp_1_3m(bx, ax, h1x, y[j], func)

        else:

            integr_y += (h1x * h1y / 9) * simp_1_3m(bx, ax, h1x, y[j], func)

    return integr_y


lhs_res = d_simp_1_3(1, 0, 1, 0, 0.0001, 0.0001, func_xy)

print("Value of the Integral in LHS {}".format((lhs_res)))

t2 = time.time()

print("Time taken to execute the code {} s".format(t2 - t1))
