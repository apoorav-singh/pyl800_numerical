# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:29:45 2021

@author: Apoorav Singh Deo (Enrollment Number: 2021PHZ8046) 

Intended to compelete the course requiremnt of PYL800 - 

Numerical and Computational method in Research

Do not distribute.
"""

import random 
# Importing Random to generate random number
import math
# Importing math modules to access mathematical Functions
from matplotlib import pyplot as plt
# Importing matplotlib for plotting the graphs
import numpy as np
# Importing numpy module to access faster arrays

sgm = np.array([0.1, 0.5, 1, 5])
# Defining sigma values as mentioned in the question

#sgm = np.linspace(0.1,5,100)

ite_len = len(sgm)
# It calculates the array size of input sigma value (Helpful in iteration)




# Defining the Gaussian Function [For Debugging only]

def gaussian(sigm, x, y):

# sigm is the sigma value of the gaussian function
# x is the argument of the function

    g_f = 1/(2*math.pi*sigm**2)*math.exp(-(x*x + y*y)/(2*sigm**2))
    
# "g_f" is the return variable with parameters for gaussian function
    
    return g_f

N = 10**6
# Number of alpha-particle bobarded towards the atom
Z = 79
# Atomic number of the Gold atom [amu] 
e = 1.602e-19
# Electronic Charge [C]
E = 7.7e6*e
# Energy of incoming alpha particle [eV] ==> [J]
eps_0 = 8.854e-12
# Constant \epsilon_0 [SI]
a0 = 5.292e-11
# Bohr Radius [SI]

# Gaussian Type Random number Generator Function
def g_R(sigm):

# "sigm" is the sigma value of the Gaussian Function
 
    r = math.sqrt(-2*sigm**2*math.log(1-random.random()))
# Generation of radom radial points
    theta = 2*math.pi*random.random()
# Generation of Random \theta values 
    x = r*math.cos(theta)
# Building x - co-ordinate out of the random values
    y = r*math.sin(theta)
# Building y - co-ordinate out of the random values
    return x,y
# Return Function

x1 = np.empty([N,ite_len])
y1 = np.empty([N,ite_len])
z1 = np.empty([N,ite_len])
# Decalaring two dimensional array (z1 is declared for debugging purposes) 
# Numpy array is declared 

k = 0
# Used for indexing cnt[] 

cnt = np.empty(ite_len)
# Declaring the "cnt" variable to count the atoms reflected back


for j in sgm:
    
    print("Running simulation for Sigma value {}".format(j))
    
    cnt[k] = 0
    # Setting value to zero to avoid garbage values.

    b_par = (Z*e*e/(2*math.pi*eps_0*E))
    # Parameter declaration: Distance of Closest Approach

    #print(b_par)
    print("Generating Random Number Stream . . . ")
    for i in range(N):
        x1[i,k],y1[i,k] = g_R(j*a0/100)
        # Initializing the function for random number generation
        z1[i,k] = gaussian(j*a0/100, x1[i,k], y1[i,k])
        # For debugging purpose only
        b = math.sqrt(x1[i,k]*x1[i,k]+y1[i,k]*y1[i,k])
        # Pin pointing data value
        #print(b)
        if (b < b_par):
            cnt[k] += 1
    
    print("{} particles were reflected back out of {}".format(cnt[k], N))
    print("Simulation Done!")
    print("\n---------------------------------------------\n")
    k += 1
    # Iteration finishes
        
            

#---------------------
# Plotting
plt.style.use('seaborn')    
plt.plot(sgm, cnt/N, "r-o", label=str(N))
plt.xlabel(r"$\sigma\ \rightarrow$")
plt.ylabel(r"$Fraction\ of\ Reflected\ \alpha\ particles \rightarrow$")
plt.legend()
plt.title("Simulating Rutherford Experiment")
plt.grid(1)
plt.show()