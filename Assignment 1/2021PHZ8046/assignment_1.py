# Assignment 1 - (PYL800)
# Code By: Apoorav Singh Deo 
# En. Number: 2021PHZ8046

import math 
import numpy as np

from matplotlib import pyplot as plt

# [Parameters]
l = 89.0 # [inches]
beta_1 = 11.5*(math.pi/180) # [radians]
h = 49.0 # [inches]


# Function Implementation

def func_1(alpha, D):
    A = l*math.sin(beta_1)
    B = l*math.cos(beta_1)
    C = (h + 0.5*D)*math.sin(beta_1) - 0.5*D*math.tan(beta_1)
    E = (h + 0.5*D)*math.cos(beta_1) - 0.5*D
    y = A*math.sin(alpha)*math.cos(alpha) + B*(math.sin(alpha))**2 - C*math.cos(alpha) - E*math.sin(alpha)

    return y

# Function Derivative Implementation

def func_2(alpha, D):
    A = l*math.sin(beta_1)
    B = l*math.cos(beta_1)
    C = (h + 0.5*D)*math.sin(beta_1) - 0.5*D*math.tan(beta_1)
    E = (h + 0.5*D)*math.cos(beta_1) - 0.5*D
    y = A*math.cos(2*alpha) + B*(math.sin(2*alpha)) - C*math.sin(alpha) - E*math.sin(alpha)

    return y

# Algorithm for Bisection method

# Initial guess is taken to be as 0 and 1
print("\t Bisection Method")
print("\n")
print(r"$\theta_i = 0^o and \theta_f = 90^o$")
print("\n")

n1 = 10 # Number of Iterations for D
i = 0 # Iteration Variable 


dump_c = np.empty(n1)
# Would store the value of the last value of c i.e. c = (a+b)/2
# For each iteration 

dump_f_c = np.empty(n1)
# Would store the value of the last value of f(c)
# For each iteration

alpha = np.empty(n1)


tol = 1e-6 # Tolerace limit

D_range = np.linspace(30, 100, n1) # units [in]

for D in D_range:
    
    j = 0 # Help in breaking the loop at desired point.
    
    print("Running for D = {}".format(D))
    print("\n")
    
    a = 0*(math.pi/180)
    b = 90*(math.pi/180)

    
    
    #f_a = func_1(a, D)
    #f_b = func_1(b, D)
    
    c = (b+a)*0.5
    f_c = func_1(c, D)
    

# Redeclaration of the 'c' and 'f_c' is done to compute the values inside the loop
    
    while (abs(f_c) >= tol):
        
        c = abs((b+a)*0.5)
        f_c = func_1(c, D)
        
        f_a = func_1(a, D)
        f_b = func_1(b, D)
        
        if ((f_a*f_c)<0 and (f_b*f_c)>0):
            b = c
        
            
        elif ((f_a*f_c)>0 and (f_b*f_c)<0):
            a = c
        
        if (j == 10):
            break
        
        j += 1 
        #print(c*(180/math.pi),f_c)
    
# Algorithm for Newton-Raphson Method
    
    while (abs(f_c) >= tol):
        
        x = c - (func_1(c, D)/func_2(c, D))
        c = x
        f_c = func_1(c, D)
        print(c*(180/math.pi),f_c)
        
                
            
    #print(k)
    dump_f_c[i] = f_c
    dump_c[i] = c    
    
    alpha[i] = (180/math.pi)*dump_c[i]
    
    i += 1
    print("\n")
        
plt.plot(alpha, D_range, marker='o', label=r'Tol $\approx 10^{-6}$')
plt.legend()
plt.xlabel(r"$\alpha \rightarrow$")
plt.ylabel(r"$D\ \rightarrow$")
plt.grid(True)
plt.title("Bisection Method + Newton-Raphson Method")
plt.grid(which='minor')
plt.show()

plt.savefig("Bisection_Method.png", dpi=900)
    
