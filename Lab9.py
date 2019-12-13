# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:18:23 2019

@author: holt3393
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
##################################
##### TIME ARRAY DEFINITION ######
##################################
steps = 1e-2
tmin = 0
tmax = 2
t = np.arange(tmin,tmax,steps)
##################################

def my_fft(x, fs):
    N = len(x)
    X_fft = fft.fft(x)
    X_fft_shifted = fft.fftshift(X_fft)
    
    freq = np.arange(-N/2,N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return X_mag, X_phi, freq


def totalplot(x, t ,funcname, lower, upper):
    
    X_mag, X_phi, freq = my_fft(x, 1000)
    plt.figure(figsize=(10,7))
    plt.subplot(3,1,1) #Top Figure
    plt.plot(t,x)
    plt.title('FFT of ' + funcname)
    plt.ylabel(funcname)
    plt.xlabel("t (s)")

    plt.subplot(3,2,3) #Mag 1
    plt.stem(freq, X_mag)
    plt.ylabel("Magnitude")

    plt.subplot(3,2,4)#Mag 2
    plt.stem(freq, X_mag)
    plt.xlim([lower,upper])

    plt.subplot(3,2,5)#Phase 1
    plt.stem(freq, X_phi)
    plt.ylabel("Phase")
    plt.xlabel("Freq. (Hz)")

    plt.subplot(3,2,6)#Phase 2
    plt.stem(freq, X_phi)
    plt.xlim([lower,upper])
    plt.xlabel("Freq. (Hz)")
    plt.show()
    
def f1(t):
    return np.cos(2*np.pi*t)

def f2(t):
    return 5*np.sin(2*np.pi*t)

def f3(t):
    x=2*np.cos((4*np.pi*t)-2)+(np.sin((12*np.pi*t)+3)*np.sin((12*np.pi*t)+3))
    return x

def b(k):
    b = (-2*np.cos(np.pi*k) + np.cos(2*np.pi*k) + 1)/(np.pi*k)
    return b

def FSapprox(n,t,T):
    x=0
    for i in range(1,n+1):
        x = x + b(i)*np.sin(i*(2*np.pi/T)*t)
    return x

totalplot(f1(t), t, "f1", -2, 2)
totalplot(f2(t), t, "f2", -2, 2)
totalplot(f3(t), t, "f3", -15, 15)

def my_fft_clean(x, fs):
    N = len(x)
    X_fft = fft.fft(x)
    X_fft_shifted = fft.fftshift(X_fft)
    
    freq = np.arange(-N/2,N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(N):
        if (np.abs(X_mag[i]) < 1e-10):
            X_phi[i] = 0
    return X_mag, X_phi, freq

def totalplotclean(x, t ,funcname, lower, upper):
    
    X_mag, X_phi, freq = my_fft_clean(x, 100)
    plt.figure(figsize=(10,7))
    plt.subplot(3,1,1) #Top Figure
    plt.plot(t,x)
    plt.title('Clean FFT of ' + funcname)
    plt.ylabel(funcname)
    plt.xlabel("t (s)")

    plt.subplot(3,2,3) #Mag 1
    plt.stem(freq, X_mag)
    plt.ylabel("Magnitude")

    plt.subplot(3,2,4)#Mag 2
    plt.stem(freq, X_mag)
    plt.xlim([lower,upper])

    plt.subplot(3,2,5)#Phase 1
    plt.stem(freq, X_phi)
    plt.ylabel("Phase")
    plt.xlabel("Freq. (Hz)")

    plt.subplot(3,2,6)#Phase 2
    plt.stem(freq, X_phi)
    plt.xlim([lower,upper])
    plt.xlabel("Freq. (Hz)")
    plt.show()

totalplotclean(f1(t), t, "f1", -2, 2)
totalplotclean(f2(t), t, "f2", -2, 2)
totalplotclean(f3(t), t, "f3", -15, 15)
##################################
##### TIME ARRAY DEFINITION ######
##################################
steps = 1e-2
tmin = 0
tmax = 16
t = np.arange(tmin,tmax,steps)
##################################
totalplotclean(FSapprox(15,t,8), t, "Square Fourier", -2, 2)