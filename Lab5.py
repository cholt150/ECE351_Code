#LAB 5
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

##################################
steps = 1e-5
t = np.arange(0,1.2e-3+steps,steps) #DEFINE TIME ARRAY

R = 1000
L = 27e-3
C = 100e-9

def toRad(deg):
    return ((deg * np.pi)/180)

def h(t):
    h = 10331 * np.exp(-5000*t)*np.sin(18584*t+toRad(105))*u(t)
    return h

def hstep(t):
    h = (1/18584) * np.exp(-5000*t)*np.sin(18584*t)*u(t)
    return h

#STEP FUNCTION
def u(t):
    y = np.zeros((len(t)))
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

numH = [(1/(R*C)),0]
demH = [1,(1/(R*C)),(1/(L*C))]

tout, yout = sig.impulse((numH,demH), T = t)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1) 
plt.plot(t,h(t))
plt.title('Impulse Response of RLC Circuit') 
plt.ylabel('Hand-Solved Response') 
plt.grid(True)

plt.subplot(2,1,2) 
plt.plot(tout,yout)
plt.ylabel('Scipy Impulse Response') 
plt.grid(True)
plt.xlabel('t')
plt.show() 

tstep,ystep = sig.step((numH,demH), T = t)

plt.figure(figsize = (12,8))
plt.plot(tstep,ystep)
plt.title('Step Response of RLC Circuit') 
plt.ylabel('sig.step Response') 
plt.grid(True)
plt.xlabel('t')
plt.show() 

plt.figure(figsize = (12,8))
plt.plot(t,hstep(t))
plt.title('Impulse Response of RLC Circuit') 
plt.ylabel('Hand-Solved Response') 
plt.grid(True)
plt.xlabel('t')
plt.show() 