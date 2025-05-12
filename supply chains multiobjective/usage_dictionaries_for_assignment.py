#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 08:58:51 2023

@author: whadyimac
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

#mapdict={0:tuple([0,1]),1:tuple([2,7]),3:tuple([4,6])}
#imapdict= {value: key for key, value in mapdict.items()}

ni=3 #sources
nj=3 #sinks

source_avai=np.zeros(ni)
sink_dem=np.zeros(nj)
source_avai=[20,55,60]
sink_dem=[10,11,12]


k=0
dmap={}
for i in range(ni):
    for j in range(nj):
        dmap[k]=(i,j)
        k+=1
idmap= {value: key for key, value in dmap.items()}        
        
def consfun_eq(x):
    y=np.zeros(3)
    y[0]=x[idmap[(0,0)]]+x[idmap[(1,0)]]+x[idmap[(2,0)]]
    y[1]=x[idmap[(0,1)]]+x[idmap[(1,1)]]+x[idmap[(2,1)]]
    y[2]=x[idmap[(0,2)]]+x[idmap[(1,2)]]+x[idmap[(2,2)]]  
    return y

def consfun_ueq(x):
    y=np.zeros(3)
    y[0]=x[idmap[(0,0)]]+x[idmap[(0,1)]]+x[idmap[(0,2)]]
    y[1]=x[idmap[(1,0)]]+x[idmap[(1,1)]]+x[idmap[(1,2)]]
    y[2]=x[idmap[(2,0)]]+x[idmap[(2,1)]]+x[idmap[(2,2)]]    
    return y

def fun(x):
    P=x[idmap[(0,0)]]+x[idmap[(0,1)]]+x[idmap[(0,2)]]+\
    2*(x[idmap[(1,0)]]+x[idmap[(1,1)]]+x[idmap[(1,2)]])+\
    1.2*(x[idmap[(2,0)]]+x[idmap[(2,1)]]+x[idmap[(2,2)]])  
    return P

bounds=[]
ub_eq=[]
lb_eq=[]
ub_ueq=[]
lb_ueq=[]

for i in range(9):
    bounds.append((0,None))

for i in range(3):
    ub_eq.append(sink_dem[i])
    lb_eq.append(sink_dem[i])
for i in range(3):
    ub_ueq.append(source_avai[i])
    lb_ueq.append(0)
        
 
x0=np.ones(9)    
constraints_eq= NonlinearConstraint(consfun_eq, lb_eq, ub_eq)
constraints_ueq= NonlinearConstraint(consfun_ueq, lb_ueq, ub_ueq)
constr_list=[constraints_eq,constraints_ueq]
options={'ftol':1e-6}
result=minimize(fun, x0,method='SLSQP', bounds=bounds,constraints=constr_list)
x=result.x  
  
F=np.zeros((3,3))
for i in range(3):
    for j in range(3):
        F[i,j]=x[idmap[(i,j)]]
        
print(result.message)
print(result.success)
print('fun=',result.fun) 
print('Flows:',F)       


print([np.sum(F[0,:]),np.sum(F[1,:]),np.sum(F[2,:])])
print([np.sum(F[:,0]),np.sum(F[:,1]),np.sum(F[:,2])])
print(x)
        


