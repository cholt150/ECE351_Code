# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:20:14 2019
Lab 6
@author: Cory Holt
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
##################################
##### TIME ARRAY DEFINITION ######
##################################
steps = 1e-5
tmin = 0
tmax = 2
t = np.arange(tmin,tmax+steps,steps)
##################################
########## PART 1 ################
def u(t):
    y = np.zeros((len(t)))
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def toRad(deg):
    return ((deg * np.pi)/180)

def y(t):
    y = (np.exp(-6*t)-.5*np.exp(-4*t)+.5)*u(t)
    return y

numH = [1,6,12]
demH = [1,10,24]

tout, yout = sig.step((numH,demH), T=t)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1) 
plt.plot(t,y(t))
plt.title('Step Response of Differential Equation') 
plt.ylabel('Hand-Solved Response') 
plt.grid(True)

plt.subplot(2,1,2) 
plt.plot(tout,yout)
plt.ylabel('Scipy Step Response') 
plt.grid(True)
plt.xlabel('t')
plt.show() 

demH = [1,10,24,0]

R, P, K = sig.residue(numH,demH)

print("R,P,K for part 1:\n")
print(R)
print(P)
print(K)

##################################
########## PART 2 ################

##################################
##### TIME ARRAY DEFINITION ######
##################################
steps = 1e-5
tmin = 0
tmax = 4.5
t = np.arange(tmin,tmax+steps,steps)
##################################

dem2 = [1,18,218,2036,9085,25250,0]
num2 = [25250]

R2,P2,K2 = sig.residue(num2,dem2)

print("R,P,K for part 2:\n")
print(R2)
print(P2)
print(K2)

def cosMethod(R,P,t):
    y = 0
    for i in range(len(R)):
        y += (abs(R[i])*np.exp(np.real(P[i])*t)*np.cos(np.imag(P[i]*t)+np.angle
        (R[i]))*u(t))
    return y

y2 = cosMethod(R2,P2,t)

numStep = [25250]
demStep = [1,18,218,2036,9085,25250]

tout,yout = sig.step((numStep,demStep), T = t)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1) 
plt.plot(t,y2)
plt.title('Step Response of Much More Complex Differential Equation') 
plt.ylabel('Cosine Method') 
plt.grid(True)

plt.subplot(2,1,2) 
plt.plot(tout,yout)
plt.ylabel('Scipy Step Response') 
plt.grid(True)
plt.xlabel('t')
plt.show() 