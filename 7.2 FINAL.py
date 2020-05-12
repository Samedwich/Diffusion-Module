# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:47:25 2020

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt
import random 

n=10000
tmax=1000
pi=np.pi
x=np.zeros(tmax)
y=np.zeros(tmax)
xavg=np.zeros(tmax)
yavg=np.zeros(tmax)
tindex=np.zeros(tmax)
rms=np.zeros(tmax)

def direc():
    return random.uniform(0,2*pi)
def step():
    return np.random.gamma(2,1)
for i in range(tmax):
    tindex[i]=i
for i in range(n):
    for k in range(1,tmax):
        
        tempangle = direc()
        x[k]= x[k-1]+ (step()*np.cos(tempangle))
        y[k]= y[k-1]+ (step()*np.sin(tempangle))
        xavg[k]+=x[k]
        yavg[k]+=y[k]
        
        rms[k]+=(x[k]**2 + y[k]**2)**0.5
  
xavg=xavg/n
yavg=yavg/n
rms=rms/(n)

(m,b)=np.polyfit(np.log(tindex[1:]), np.log(rms[1:]), 1)
r=np.exp(b)

model = r*(tindex**m)
res=rms-model


fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(221)
plt.plot(tindex, xavg,label="Average x Position")
plt.plot(tindex,yavg,label="Average y Position")
plt.xlabel("Time (step)")
plt.ylabel("Average Position")
plt.legend(loc="lower left")
plt.title("Average x-Position vs. Time")


ax3 = fig.add_subplot(222)
plt.plot(tindex,rms, linewidth=5, label="Average radius")
plt.plot(tindex, model, color = "red", label="Best fit")
plt.xlabel("Time(step)")
plt.ylabel("Average Radius")
plt.legend(loc="lower right")
plt.title("RMS Distance vs. Time")

ax4 = fig.add_subplot(223)
plt.plot(tindex, res)
plt.xlabel("Time (step)")
plt.ylabel("Residual")
plt.title("Residual vs. Time ")

plt.show()   