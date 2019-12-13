"""
Created on Tue Oct 15 18:29:57 2019
    
@author: holt3393
"""
import matplotlib.pyplot as plt
import scipy.signal as sig

NumG = [1,9]
DenG = [1,-2,-40,-64]

NumA = [1,4]
DenA = [1,4,3]

NumB = [1,26,168]
DenB = [1]

Z, P, K = sig.tf2zpk(NumG,DenG)

print("G(s):")
print("Zeros: ", Z)
print("Poles: ", P)
print("Gain:  ", K)
print("\n")

Z, P, K = sig.tf2zpk(NumA,DenA)

print("A(s):")
print("Zeros: ", Z)
print("Poles: ", P)
print("Gain:  ", K)
print("\n")

Z, P, K = sig.tf2zpk(NumB,DenB)

print("B(s):")
print("Zeros: ", Z)
print("Poles: ", P)
print("Gain:  ", K)
print("\n")

### For Open Loop Tranfer

NumH = sig.convolve(NumA,NumG)
DenH = sig.convolve(DenA,DenG)

Z, P, K = sig.tf2zpk(NumH,DenH)

print("Open Loop Transfer Function:")
print("Zeros: ", Z)
print("Poles: ", P)
print("Gain:  ", K)
print("\n")

print("Open Loop Transfer Function:")
print("Numerator: ", NumH)
print("Denominator: ", DenH)

tout, yout = sig.step((NumH,DenH))
plt.figure(figsize=(12,8))
#plt.subplot(2,1,1) 
plt.plot(tout,yout)
plt.title('Step Response of Open Loop Transfer fn') 
plt.ylabel('Step Response') 
plt.grid(True)
plt.show()

NumClose = sig.convolve(NumA,NumG)
DenClose = sig.convolve(DenA,(DenG + sig.convolve(NumG,NumB)))

Z, P, K = sig.tf2zpk(NumClose,DenClose)

print("Closed Loop Transfer Function:")
print("Zeros: ", Z)
print("Poles: ", P)
print("Gain:  ", K)
print("\n")

print("Closed Loop Transfer Function:")
print("Numerator: ", NumClose)
print("Denominator: ", DenClose)

tout, yout = sig.step((NumClose,DenClose))
plt.figure(figsize=(12,8))
#plt.subplot(2,1,1) 
plt.plot(tout,yout)
plt.title('Step Response of Closed Loop Transfer fn') 
plt.ylabel('Step Response') 
plt.grid(True)
plt.show()