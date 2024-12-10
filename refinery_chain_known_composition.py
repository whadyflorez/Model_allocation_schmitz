#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:40:43 2023

@author: whadyimac
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

def fun(x,*args):
    nf=args[0]
    ng=args[1]
    F=x[0:2]
    G=x[2:4]
    totF=np.sum(F)
    totG=np.sum(G)
    P=nf*totF+ng*totG  
#    P=F[0]+2*F[1]+3*G[0]+G[1]
    return -P

#availability of resources F and G at the source
SF=np.array([3,5])
SG=np.array([3,5])

#required fuel mix composition
comp=(0.3,0.7)

n=4 #number of flows (allocations)
#initial guesses and array mapping into x
F0=np.array([1,1])
G0=np.array([1.5,1.7])
x0=np.array([F0[0],F0[1],G0[0],G0[1]])

bnds=((0,SF[0]),(0,SF[1]),(0,SG[0]),(0,SG[1]))
optns={'disp':True}
def consfun(x):
    F=x[0:2]
    G=x[2:4]
#    y=np.zeros(2)
    y=np.zeros(1)
    totF=np.sum(F)
    totG=np.sum(G)
    totmix=totF+totG
    y[0]=totF/totmix
#    y[1]=totG/totmix
    return y
tolbdns=1e-3
#lb=[comp[0]-tolbdns,comp[1]-tolbdns]
#ub=[comp[0]+tolbdns,comp[1]+tolbdns]
lb=[comp[0]]
ub=[comp[0]]
nlc= NonlinearConstraint(consfun, lb, ub)
result=minimize(fun, x0, args=(comp[0],comp[1]),method='SLSQP', bounds=bnds,constraints=nlc,\
options=optns)

F=result.x[0:2]
G=result.x[2:4]  
print('F:',F)
print('G:',G)
print('Flow_max=',-result.fun)
print(np.sum(F)/(np.sum(F)+np.sum(G)),np.sum(G)/(np.sum(F)+np.sum(G)))



