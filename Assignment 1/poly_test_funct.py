import math 
import numpy as np

from matplotlib import pyplot as plt



l = 89.0 # [inches]
beta_1 = 11.5*(math.pi/180) # [radians]
h = 49.0 # [inches]

# Function definition is changed for the implementation in the code
# sin(x) is changed with t. This converts the whole equation into a ploynomial from a trignometric equation.
# Squaring is done to remove the square roots.

def func_1(t, D):
    A = l*math.sin(beta_1)
    B = l*math.cos(beta_1)
    C = (h + 0.5*D)*math.sin(beta_1) - 0.5*D*math.tan(beta_1)
    E = (h + 0.5*D)*math.cos(beta_1) - 0.5*D
    y = -(A**2 + B**2)*t**4 + (2*A*C+2*B*E)*t**3 + (-C**2-E**2+A**2)*t**2 - 2*A*C*t + C**2 

    return y

n = 1000

x = np.linspace(0.2,0.6,n)

y = np.empty(n)

i = 0

for i in range(n):
    y[i] = func_1(x[i], 30)
    
    if (abs(y[i]) <= 1e-4):
        alpha_z = x[i]
        print("The angle where function falls zero is alpha= {} ".format(x[i]*(180/math.pi)))

plt.plot(x*(180/math.pi), y, label='30 [in]')
plt.legend()
plt.grid(True)
plt.xlabel(r"$\alpha^o$")
plt.ylabel("Test Polynomial function")
plt.title("The angle where function falls zero is alpha= {} ".format(alpha_z*(180/math.pi)))