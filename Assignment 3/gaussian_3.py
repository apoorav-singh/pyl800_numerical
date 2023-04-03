from math import *
from random import random 
from matplotlib import pyplot as plt

# Constants
Z = 79
e = 1.602e-19
E = 7.7e6*e
epsilon0 = 8.854e-12
a0 = 5.292e-11
sigma = a0/100

N = 1000000
inp_val = [0.1, 0.5, 1, 5]
count = [0, 0, 0, 0]
j = 0 # loop
for sc in inp_val:

    # Function to generate two Gaussian random numbers
    def gaussian():
        r = sqrt (-2*sc*sc*sigma*sigma*log(1-random()))
        theta = 2*pi*random()
        x = r*cos(theta)
        y = r*sin(theta)
        return x,y

    # Main program
    count[j] = 0
    for i in range(N):
        x,y = gaussian()
        b = sqrt (x*x+y*y)
        if b<Z*e*e/(2*pi*epsilon0*E):
            count[j] += 1

    print(count[j],"particles were reflected out of",N)
    j += 1


plt.plot(count, inp_val, "b-o")
plt.xlabel("N")
plt.ylabel("Sigma")
plt.grid(1)
plt.show()
