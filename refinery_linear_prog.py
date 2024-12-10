#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:40:43 2023

@author: whadyimac
"""
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import linprog_verbose_callback


ni=3  #number of sources
nc=3  #number of components
nf=ni*nc  #number of flows to allocate

#availability of reach source esources 
SF=np.zeros((ni,nc))
SF[0,:]=[1,2.5,3.7]
SF[1,:]=[4,5.1,6]
SF[2,:]=[3,5,1.2]

#unit cost of each flow
CF=np.zeros((ni,nc))
CF[0,:]=[1,1,1]
CF[1,:]=[2,2,2]
CF[2,:]=[3,3,3]

#required fuel mix composition
comp=np.array([0.3,0.5,0.2])

#mapping F[0,0:nc]=x[0:nc],F[1,0:nc]=x[nc:2*nc],F[2,0:nc]=x[2*nc:3*nc]....
#F[i,0:nc]=x[i*nc:(i+1)*nc] i=0:ni
c=np.zeros(nf)
#c[0:nc]=comp
#c[nc:2*nc]=comp
#c[2*nc:3*nc]=comp
#for i in range(ni):
#    c[i*nc:(i+1)*nc]=comp

#maximization of product flow
# c[0]=comp[0]
# c[1]=comp[1]
# c[2]=comp[2]
# c[3]=comp[0]
# c[4]=comp[1]
# c[5]=comp[2]
# c[6]=comp[0]
# c[7]=comp[1]
# c[8]=comp[2]
# c=-c #maximization

#maximization of costs (minimum cost is 0 and implies 0 flows under ordinary conditions. 
#Paradox of the imbecile manager)
c[0]=CF[0,0]
c[1]=CF[0,1]
c[2]=CF[0,2]
c[3]=CF[1,0]
c[4]=CF[1,1]
c[5]=CF[1,2]
c[6]=CF[2,0]
c[7]=CF[2,1]
c[8]=CF[2,2]
#c=-c



bnds=[] 
bnds.append((0,SF[0,0]))
bnds.append((0,SF[0,1]))
bnds.append((0,SF[0,2]))
bnds.append((0,SF[1,0]))
bnds.append((0,SF[1,1]))
bnds.append((0,SF[1,2]))
bnds.append((0,SF[2,0]))
bnds.append((0,SF[2,1]))
bnds.append((0,SF[2,2]))

A_eq=np.zeros((nc-1,nf))
b_eq=np.zeros(nc-1)
A_eq[0,0]=(1-comp[0])
A_eq[0,1]=-comp[0]
A_eq[0,2]=-comp[0]
A_eq[0,3]=(1-comp[0])
A_eq[0,4]=-comp[0]
A_eq[0,5]=-comp[0]
A_eq[0,6]=(1-comp[0])
A_eq[0,7]=-comp[0]
A_eq[0,8]=-comp[0]

A_eq[1,0]=-comp[1]
A_eq[1,1]=(1-comp[1])
A_eq[1,2]=-comp[1]
A_eq[1,3]=-comp[1]
A_eq[1,4]=(1-comp[1])
A_eq[1,5]=-comp[1]
A_eq[1,6]=-comp[1]
A_eq[1,7]=(1-comp[1])
A_eq[1,8]=-comp[1]

b_eq[0]=0 
b_eq[1]=0 

#constraint to solve the stupid manager paradox
A_ub=np.zeros((1,nf))
b_ub=np.zeros(1)
A_ub[0,0]=-comp[0]
A_ub[0,1]=-comp[1]
A_ub[0,2]=-comp[2]
A_ub[0,3]=-comp[0]
A_ub[0,4]=-comp[1]
A_ub[0,5]=-comp[2]
A_ub[0,6]=-comp[0]
A_ub[0,7]=-comp[1]
A_ub[0,8]=-comp[2]

b_ub[0]=-6.0
    


result=linprog(c, A_ub=A_ub,b_ub=b_ub,A_eq=A_eq, b_eq=b_eq, bounds=bnds, method='simplex')


x=result.x
fun=result.fun
status=result.status
success=result.success

F=np.zeros((3,3))
F[0,0:3]=x[0:3]
F[1,0:3]=x[3:6]
F[2,0:3]=x[6:9]

print('F:',F)
print('fun=',fun)
print(result.message)
print('status=',status)
print('success=',success)





