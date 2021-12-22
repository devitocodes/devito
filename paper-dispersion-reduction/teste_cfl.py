#==============================================================================
import numpy as np
import sys
import math as mt
#==============================================================================

#==============================================================================
teste = 1

if(teste==1):
    
    dx     = 20 
    vmax   = 3.0
    dt_ref = 0.1
    tn     = 2000
    t0     = 0

if(teste==2):
    
    dx     = 20 
    vmax   = 4.5
    dt_ref = 0.1
    tn     = 4000
    t0     = 0

if(teste==3):
    
    dx     = 10 
    vmax   = 3.0
    dt_ref = 0.1
    tn     = 2000
    t0     = 0

if(teste==4):
    
    dx     = min(50,32) 
    vmax   = 7.0
    dt_ref = 0.1
    tn     = 3000
    t0     = 0
#==============================================================================

#==============================================================================
step     = 0.0001
CFL_min  = step
CFL_max  = 0.7

vcfl     = np.arange(CFL_min,CFL_max,step)
namostra = vcfl.shape[0]

for i in range(0,namostra):
    
    factor = 10
    CFL    = vcfl[i]
    dt     = (dx*CFL)/vmax
    
    ntmax  = int((tn-t0)/dt)
    dt0    = (tn-t0)/(ntmax)
    nt     = int((tn-t0)/dt0)
    nplot  = 21
    jump   = mt.ceil(nt/(nplot-1)) 
    
    rref   = dt/dt_ref 
    test_number =  (factor*dt0)//1
    
    if(test_number==(factor*dt0)):     
        print('CFL: %f - dt: %f - jump: %d - nt: %d - ration: %f'%(CFL,dt0,jump,nt,rref))
#==============================================================================