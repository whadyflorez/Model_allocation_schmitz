#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:00:12 2023

@author: whadyimac
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

def fun(x,*args):
    pf=500
    pg=1000
    totF=x[0]+x[1]
    totG=x[2]+x[3]
    totmix=totF+totG
    P=pf*totF/totmix+pg*totG/totmix
    return -P

SF0=5
SF1=2
SG0=3
SG1=1

n=4
x0=np.array([1,1,1,1])

bnds=((0,SF0),(0,SF1),(0,SG0),(0,SG1))
optns={'disp':True}
def consfun(x):
    y=np.zeros(2)
    totF=x[0]+x[1]
    totG=x[2]+x[3]
    totmix=totF+totG
    y[0]=totF/totmix
    y[1]=totG/totmix
    return y
lb=[0.1,0.1]
ub=[1,1]  
nlc= NonlinearConstraint(consfun, lb, ub)
result=minimize(fun, x0, method='SLSQP', bounds=bnds,constraints=nlc,\
options=optns)

print('F:',result.x[0:2])
print('G:',result.x[2:4])
print('Pmax=',-result.fun)


