# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:16:14 2019

@author: holt3393
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

R = 1000
L = 27e-3
C = 100e-9

steps = 1000
w = np.arange(1e3,1e6+steps,steps)
##PART 1-1
magH = 20*np.log10((w/(R*C))/(np.sqrt(((1/(L*C))-w**2)**2 + (w/(R*C))**2)))

phaseH = ((np.pi/2)-np.arctan((w/(R*C))/(-w**2 + 1/(L*C)))) * 180/np.pi
for i in range(len(phaseH)):
    if (phaseH[i] > 90):
        phaseH[i] = phaseH[i] - 180

plt.figure(figsize=(10,7))
plt.subplot(2,1,1) #Top Figure
plt.grid()
plt.semilogx(w, magH)
plt.title("hand-solved")
plt.subplot(2,1,2)
plt.grid()
plt.semilogx(w, phaseH)
plt.show()

##PART 1-2
numh = [1/(R*C),0]
denh = [1,1/(R*C),1/(L*C)]
sys = sig.TransferFunction(numh,denh)
ang,mag,phase = sig.bode(sys,w)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1) #Top Figure
plt.grid()
plt.semilogx(ang, mag)
plt.title("sig.bode")
plt.subplot(2,1,2)
plt.grid()
plt.semilogx(ang, phase)
plt.show()

sys = con.TransferFunction(numh,denh)
_ = con.bode(sys, w, dB=True, deg=True, Plot=True)

t_steps = 1e-7
t = np.arange(0, 0.01+t_steps, t_steps)
x = (np.cos(2*np.pi*100*t)+ np.cos(2*np.pi*3024*t) + np.sin(2*np.pi* 50000*t))

numZ, denZ = sig.bilinear(numh, denh, 1/t_steps)
xFiltered = sig.lfilter(numZ, denZ, x)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1) #Top Figure
plt.grid()
plt.plot(t,x)
plt.title("Unfiltered Signal")
plt.subplot(2,1,2)
plt.plot(t,xFiltered)
plt.title("Filtered Signal")
plt.grid()
#plt.semilogx(ang, phase)
plt.show()