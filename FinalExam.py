# -*- coding: utf-8 -*-
"""
Holt, Cory
151-23926
Section 51
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                  24,25])
arange = np.arange(1,26,1)
linspace = np.linspace(1,25,25)

"""
I'm not sure if this is what you intended with np.array, but this is the first 
time I'd used it in a program.
np.array creates a numpy array for use with numpy functions out of a standard 
    array. It can specify whether or not to be a row or column vector.
np.arange creates an array from a lower bound to an upper bound(non-inclusive)
    with a given step size.
np.linspace creates an array linearly spaced array with upper and lower bounds,
    and a specified number of steps.
"""

print("np array: ", array)
print("arange array: ", arange)
print("linspace array: ",linspace)

#Time Array Definition
steps = 1e-2
t = np.arange(0,2+steps,steps)

#signal definition
def x1(t):
    return 4*np.cos(2*np.pi*3*t)

def x2(t):
    return x1(t) + 2*np.cos(2*np.pi*t)

def x3(t):
    return 3*np.exp(-6*t)*np.sin(2*np.pi*5*t)

def x4(t):
    return np.exp(-t)+2*np.exp(-2*t)+4*np.exp(-4*t)

#Figure plotting
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.plot(t,x1(t))
plt.title("x1")
plt.subplot(2,2,2)
plt.plot(t,x2(t))
plt.title("x2")
plt.subplot(2,2,3)
plt.plot(t,x3(t))
plt.title("x3")
plt.subplot(2,2,4)
plt.plot(t,x4(t))
plt.title("x4")
plt.show()

numh = [1,0,4,-18]
denh = [1,6,8,128,380]

Z, P, K = sig.tf2zpk(numh,denh)

print("H(s)")
print("Zeros: ", Z)
print("Poles: ", P)
print("\n")

tstep, ystep = sig.step((numh,denh), T=t)
timp, yimp = sig.impulse((numh,denh), T=t)

plt.figure(figsize=(10,8))
plt.plot(tstep,ystep,label = 'Step Response')
plt.plot(timp,yimp, label = 'Impulse Response')
plt.grid()
plt.xlabel("t")
plt.title("Impulse and Step Response")
plt.legend(loc="lower left")
plt.show()
"""
From the plots you can clearly see that both responses are unstable. They are 
"undamped" and continue to increase in magnitude as time increases.
"""
c1 = sig.convolve(x1(t),x4(t))
c2 = sig.convolve(c1,x2(t))
c3 = sig.convolve(c2,x3(t))

t2 = np.arange(2*t[0],2*t[len(t)-1]+steps,steps)
t3 = np.arange(3*t[0],3*t[len(t)-1]+steps,steps)
t4 = np.arange(4*t[0],4*t[len(t)-1]+steps,steps)

plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.plot(t2,c1,label = 'c1')
plt.title("f1")
plt.xlabel("t")
plt.subplot(1,3,2)
plt.plot(t3,c2, label = 'c2')
plt.title("f2")
plt.xlabel("t")
plt.subplot(1,3,3)
plt.plot(t4,c3,label='c3')
plt.title("f3")
plt.xlabel("t")
plt.show()