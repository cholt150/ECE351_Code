"""
Created on Tue Oct 22 19:14:45 2019
ECE 350 LAB 8
@author: holt3393
"""
import numpy as np
import matplotlib.pyplot as plt
##################################
##### TIME ARRAY DEFINITION ######
##################################
steps = 1e-2
tmin = 0
tmax = 20
t = np.arange(tmin,tmax+steps,steps)
##################################

def a(k):
    a = (2*np.sin(np.pi*k)-np.sin(2*np.pi*k))/(np.pi*k)
    return np.round(a,2)

def b(k):
    b = (-2*np.cos(np.pi*k) + np.cos(2*np.pi*k) + 1)/(np.pi*k)
    return b

def FSapprox(n,t,T):
    x=0
    for i in range(1,n+1):
        x = x + b(i)*np.sin(i*(2*np.pi/T)*t)
    return x

n = np.arange(0,5)
a_k = a(n)
b_k = b(n)

print("a_1 = ", a_k[1])
print("b_1 = ", b_k[1])
print("b_2 = ", b_k[2])
print("b_3 = ", b_k[3])

f1 = FSapprox(1,t,8)
f3 = FSapprox(3,t,8)
f15 = FSapprox(15,t,8)

N = [1,3,15,50,150,1500]
count = 0 #counter for N indexing
for i in [1,2]: #PLOT
    for j in [1,2,3]: #SUBPLOT
        plt.figure(i,figsize=(12,8))
        plt.subplot(3,1,j)
        plt.plot(t,FSapprox(N[count],t,8))
        plt.grid()
        plt.ylabel('N = %i' % N[count])
        if count == 0 or count == 3:
            plt.title('Fourier Series Approximations of Square Wave')
        if count == 2 or count == 5:
            plt.xlabel('t')
        plt.show()
        count += 1

