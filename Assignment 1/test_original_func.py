import math 
import numpy as np

from matplotlib import pyplot as plt



l = 89.0 # [inches]
beta_1 = 11.5*(math.pi/180) # [radians]
h = 49.0 # [inches]

def func_1(alpha, D):
    A = l*math.sin(beta_1)
    B = l*math.cos(beta_1)
    C = (h + 0.5*D)*math.sin(beta_1) - 0.5*D*math.tan(beta_1)
    E = (h + 0.5*D)*math.cos(beta_1) - 0.5*D
    y = A*math.sin(alpha)*math.cos(alpha) + B*(math.sin(alpha))**2 - C*math.cos(alpha) - E*math.sin(alpha)

    return y

def func_2(alpha, D):
    A = l*math.sin(beta_1)
    B = l*math.cos(beta_1)
    C = (h + 0.5*D)*math.sin(beta_1) - 0.5*D*math.tan(beta_1)
    E = (h + 0.5*D)*math.cos(beta_1) - 0.5*D
    y = A*math.cos(2*alpha) + B*(math.sin(2*alpha)) - C*math.sin(alpha) - E*math.sin(alpha)

    return y

n = 50

x = np.linspace(0,2*math.pi,n)

y = np.empty(n)

i = 0

for i in range(n):
    y[i] = func_1(x[i], 100)
    
    if (abs(y[i]) <= 1e-2):
        alpha_z = x[i]
        print("The angle where function falls zero is alpha= {} ".format(x[i]*(180/math.pi)))

plt.plot(x*(180/math.pi), y, label='30 [in]', marker='o')
plt.legend()
plt.grid(True)
plt.xlabel(r"$\alpha^o\ \rightarrow$")
plt.ylabel(r"$f(\alpha)\ \rightarrow$")
#plt.title("The angle where function falls zero is alpha= {} ".format(alpha_z*(180/math.pi)))

plt.savefig("func_2.png", dpi=900)