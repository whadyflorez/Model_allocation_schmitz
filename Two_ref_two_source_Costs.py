#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  30 08:58:51 2023

@author: simonschmitz
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

number_source=2 #sources
number_sink=2 #sinks
number_product=2 #products

# Define and create availability and demand
source_avai=np.array([[400,500], # availabilty for source 1 for both products
                    [401,800]]) # availability for source 2

costs = np.array([10,9]) # cost for each source

# Define dictionaries for assignment (i,j,k) -> i sink, j source, k product
b=0
dmap={} # direct map
for i in range(number_source):
    for j in range(number_sink):
        for k in range(number_product): 
            dmap[b]=(i,j,k)
            b+=1        
idmap= {value: key for key, value in dmap.items()}  # inverse map  

# required fuel mix composition
comp=np.array([0.6,0.4])   

# Define the restrictions thus the right composition of products arrives at sink    
def constraint_equal(x):
    y=np.zeros(2)
    y[0]=(1-comp[0])*(x[idmap[(0,0,0)]]+x[idmap[(1,0,0)]])-comp[0]*(x[idmap[(0,0,1)]]+x[idmap[(1,0,1)]]) # the right amount of product 0 must arrive at sink 0
    #y[1]=(1-comp[1])*(x[idmap[(0,0,1)]]+x[idmap[(1,0,1)]])-comp[1]*(x[idmap[(0,0,0)]]+x[idmap[(1,0,0)]]) # the right amount of product 1 must arrive at sink 0
    y[1]=(1-comp[0])*(x[idmap[(0,1,0)]]+x[idmap[(1,1,0)]])-comp[0]*(x[idmap[(0,1,1)]]+x[idmap[(1,1,1)]]) # the right amount of product 0 must arrive at sink 1
    #y[3]=(1-comp[1])*(x[idmap[(0,1,1)]]+x[idmap[(1,1,1)]])-comp[1]*(x[idmap[(0,1,0)]]+x[idmap[(1,1,0)]]) # the right amount of product 0 must arrive at sink 1
    return y

# Define restriction that not more products than available can be sold
def constraint_inequal(x):
    y = np.zeros(5)
    y[0]=source_avai[0,0]-x[idmap[(0,0,0)]]-x[idmap[(0,1,0)]]
    y[1]=source_avai[1,0]-x[idmap[(1,0,0)]]-x[idmap[(1,1,0)]]
    y[2]=source_avai[0,1]-x[idmap[(0,0,1)]]-x[idmap[(0,1,1)]]
    y[3]=source_avai[1,1]-x[idmap[(1,0,1)]]-x[idmap[(1,1,1)]]
    y[4]=(x[idmap[(0,0,0)]]+x[idmap[(1,0,0)]]+x[idmap[(0,0,1)]]+x[idmap[(1,0,1)]]+x[idmap[(0,1,0)]]+x[idmap[(1,1,0)]]+x[idmap[(0,1,1)]]+x[idmap[(1,1,1)]])-1335

    return y

# Objective function to minimize the costs (for each product we have a spectiv)    
def obj_costs(x):
    P = costs[0]*(x[idmap[(0,0,0)]]+x[idmap[(0,1,0)]]+x[idmap[(0,0,1)]]+x[idmap[(0,1,1)]])+\
        costs[1]*(x[idmap[(1,0,0)]]+x[idmap[(1,1,0)]]+x[idmap[(1,0,1)]]+x[idmap[(1,1,1)]])
    return P

# Objective 
def obj_max_flow(x):
    F = (x[idmap[(0,0,0)]]+x[idmap[(1,0,0)]])+\
        (x[idmap[(0,0,1)]]+x[idmap[(1,0,1)]])+\
        (x[idmap[(0,1,0)]]+x[idmap[(1,1,0)]])+\
        (x[idmap[(0,1,1)]]+x[idmap[(1,1,1)]])
    return -F

bounds=[]

for i in range(number_product*number_sink*number_source):
    bounds.append((0,None))

# Initialise the start vector
x0=np.zeros(number_product*number_sink*number_source) 

# Restricctions seperatly
cons = [{'type': 'eq', 'fun': lambda x:  constraint_equal(x)},
        {'type':'ineq','fun':lambda x: constraint_inequal(x)}]

options={'ftol':1e-1}
result=minimize(obj_costs, x0, method='SLSQP', bounds=bounds,constraints=cons,options=options)
x=result.x  

print(result)
  
F=np.zeros((number_source*number_sink*number_product))
b = 0
for i in range(number_source):
    for j in range(number_sink):
        for k in range(number_product):
            F[b]=x[idmap[(i,j,k)]]
            b = b+1

print('f',F)

np.set_printoptions(suppress= True)      
print(result.message)
print(result.success)
print('fun=',result.fun) 
print('Flows:',F) 

b = 0
for i in range(number_source):
    for j in range(number_sink):
        for k in range(number_product):
            print('Flow for:')
            print('Source: ',i)
            print('Sink: ',j)
            print('Product: ',k)
            print('Flow is: ', F[b])
            b = b+1      


# print([np.sum(F[0,:]),np.sum(F[1,:]),np.sum(F[2,:])])
# print([np.sum(F[:,0]),np.sum(F[:,1]),np.sum(F[:,2])])
        


