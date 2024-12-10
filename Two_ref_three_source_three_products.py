#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 08:58:51 2023

@author: simonschmitz
"""
import numpy as np
from scipy.optimize import minimize

number_source=3 #sources
number_sink=2 #sinks
number_product=3 #products

# Define and create availability and demand
source_avai=np.array([[400,500,300], # availabilty for source 1 for all products
             [400,800,200], # availability for source 2
             [500,920,100]])  

#costs = np.array([10,12,18]) # cost for each source

dist = np.array([[324,121], # distance from source 0 to sink 0 and 1 in km
                [267,529], # distance from source 1 to sink 0 and 1
                [201,420]]) # distance from source 2 to sink 0 and 1

cost = np.array([[0.067,0.058,0.039], # price of buying from source 0 product 0 1 2
                [0.067,0.058,0.039], # price of buying from source 1 product 0 1 2
                [0.067,0.058,0.039]]) # price of buying from source 2 product 0 1 2

#cost_c = 1.2 # price of CO2 for transportation cost
cost_g = 10.9 # price of transportation fuel


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
comp=np.array([0.5,0.4,0.1])   

# Define the restrictions thus the right composition of products arrives at sink    
def constraint_equal(x):
    y=np.zeros(4)
    y[0]=(1-comp[0])*(x[idmap[(0,0,0)]]+x[idmap[(1,0,0)]]+x[idmap[(2,0,0)]])-comp[0]*(x[idmap[(0,0,1)]]+x[idmap[(1,0,1)]]+x[idmap[(2,0,1)]]+x[idmap[(0,0,2)]]+x[idmap[(1,0,2)]]+x[idmap[(2,0,2)]]) # the right amount of product 0 must arrive at sink 0
    y[1]=(1-comp[1])*(x[idmap[(0,0,1)]]+x[idmap[(1,0,1)]]+x[idmap[(2,0,1)]])-comp[1]*(x[idmap[(0,0,0)]]+x[idmap[(1,0,0)]]+x[idmap[(2,0,0)]]+x[idmap[(0,0,2)]]+x[idmap[(1,0,2)]]+x[idmap[(2,0,2)]]) # the right amount of product 1 must arrive at sink 0
    y[2]=(1-comp[0])*(x[idmap[(0,1,0)]]+x[idmap[(1,1,0)]]+x[idmap[(2,1,0)]])-comp[0]*(x[idmap[(0,1,1)]]+x[idmap[(1,1,1)]]+x[idmap[(2,1,1)]]+x[idmap[(0,1,2)]]+x[idmap[(1,1,2)]]+x[idmap[(2,1,2)]]) # the right amount of product 0 must arrive at sink 1
    y[3]=(1-comp[1])*(x[idmap[(0,1,1)]]+x[idmap[(1,1,1)]]+x[idmap[(2,1,1)]])-comp[1]*(x[idmap[(0,1,0)]]+x[idmap[(1,1,0)]]+x[idmap[(2,1,0)]]+x[idmap[(0,1,2)]]+x[idmap[(1,1,2)]]+x[idmap[(2,1,2)]]) # the right amount of product 0 must arrive at sink 1
    return y

# Define restriction that not more products than available can be sold
# with additional restricction that flow must be larger than
def constraint_inequal(x):
    y = np.zeros(10)
    y[0]=source_avai[0,0]-x[idmap[(0,0,0)]]-x[idmap[(0,1,0)]]
    y[1]=source_avai[1,0]-x[idmap[(1,0,0)]]-x[idmap[(1,1,0)]]
    y[2]=source_avai[2,0]-x[idmap[(2,0,0)]]-x[idmap[(2,1,0)]]
    y[3]=source_avai[0,1]-x[idmap[(0,0,1)]]-x[idmap[(0,1,1)]]
    y[4]=source_avai[1,1]-x[idmap[(1,0,1)]]-x[idmap[(1,1,1)]]
    y[5]=source_avai[2,1]-x[idmap[(2,0,1)]]-x[idmap[(2,1,1)]] 
    y[6]=source_avai[0,2]-x[idmap[(0,0,2)]]-x[idmap[(0,1,2)]]
    y[7]=source_avai[1,2]-x[idmap[(1,0,2)]]-x[idmap[(1,1,2)]]
    y[8]=source_avai[2,2]-x[idmap[(2,0,2)]]-x[idmap[(2,1,2)]]
    y[9]=(x[idmap[(0,0,0)]]+x[idmap[(1,0,0)]]+x[idmap[(2,0,0)]]+\
        x[idmap[(0,1,0)]]+x[idmap[(1,1,0)]]+x[idmap[(2,1,0)]]+\
        x[idmap[(0,0,1)]]+x[idmap[(1,0,1)]]+x[idmap[(2,0,1)]]+\
        x[idmap[(0,1,1)]]+x[idmap[(1,1,1)]]+x[idmap[(2,1,1)]])+\
        x[idmap[(0,0,2)]]+x[idmap[(1,0,2)]]+x[idmap[(2,0,2)]]+\
        x[idmap[(0,1,2)]]+x[idmap[(1,1,2)]]+x[idmap[(2,1,2)]] - 2499
    return y

# Objective function to minimize the costs (for each product we have a spectiv)    
# def obj_costs(x):
#     P = costs[0]*(x[idmap[(0,0,0)]]+x[idmap[(0,1,0)]]+x[idmap[(0,0,1)]]+x[idmap[(0,1,1)]])+\
#         costs[1]*(x[idmap[(1,0,0)]]+x[idmap[(1,1,0)]]+x[idmap[(1,0,1)]]+x[idmap[(1,1,1)]])+\
#         costs[2]*(x[idmap[(2,0,0)]]+x[idmap[(2,1,0)]]+x[idmap[(2,0,1)]]+x[idmap[(2,1,1)]]) 
#     return P

# Objective Function
def obj_cost_reduzir_costo(x):  
    P = 0
    for i in range(number_source):
        for j in range(number_sink):
            for k in range(number_product):
                P = P + cost_g*dist[i,j]*cost[i,k]*x[idmap[(i,j,k)]]
    return P

# Objectiv 
def obj_max_flow(x):
    F = 0
    for i in range(number_source):
        for j in range(number_sink):
            for k in range(number_product):
                F = F + x[idmap[(i,j,k)]]
    return -F

bounds=[]

for i in range(number_product*number_sink*number_source):
    bounds.append((0,None))

# Initialise the start vector
x0=np.ones(number_product*number_sink*number_source) 

# Restricctions seperatly
cons = [{'type': 'eq', 'fun': lambda x:  constraint_equal(x)},
        {'type':'ineq','fun':lambda x: constraint_inequal(x)}]

options={'ftol':1e-1}
result=minimize(obj_cost_reduzir_costo, x0, method='SLSQP', bounds=bounds,constraints=cons,options=options)
x=result.x  

print(result)
  
F=np.zeros((number_source*number_sink*number_product))
b = 0
for i in range(number_source):
    for j in range(number_sink):
        for k in range(number_product):
            F[b]=x[idmap[(i,j,k)]]
            b = b+1

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
        


