#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
import math  as mt 
#==============================================================================

#==============================================================================
# Test Parameters
#==============================================================================
ptype  = 9
ref    = 1

if(ptype==1):
    
    dx         = 20
    dz         = 20
    vmax       = 4.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = np.array([0.06,0.1,0.2,0.3,0.4,0.5])
    vdt        = np.array([0.3,0.5,1.0,1.5,2.0,2.5])
    jumpv      = np.array([500,300,150,100,75,60])
    ref_factor = 5
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = np.array([0.1])
        vdt   = np.array([0.1])
        jumpv = np.array([1500])

if(ptype==2):
    
    dx         = 10
    dz         = 10
    vmax       = 4.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = np.array([0.2,0.4,0.6,0.8,1.0,1.2])
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    jumpv      = np.array([300,150,100,75,60,50])
    ref_factor = 5
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = np.array([0.2])
        vdt   = np.array([0.1])
        jumpv = np.array([1500])

if(ptype==3):
    
    dx         = 20
    dz         = 20
    vmax       = 5.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = np.array([0.125,0.25,0.375,0.5,0.625,0.75])
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    jumpv      = np.array([300,150,100,75,60,50])
    ref_factor = 5
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = np.array([0.125])
        vdt   = np.array([0.1])
        jumpv = np.array([1500])
        
if(ptype==4):
    
    dx         = 50
    dz         = 32
    vmax       = 8.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = np.array([0.125,0.25,0.375,0.5,0.625,0.75])
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    jumpv      = np.array([300,150,100,75,60,50])
    ref_factor = 5
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = np.array([0.125])
        vdt   = np.array([0.1])
        jumpv = np.array([1500])
        
if(ptype==5):
    
    dx         = 10
    dz         = 10
    vmax       = 4.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = 2*np.array([0.06,0.1,0.2,0.3,0.4,0.5])
    vdt        = np.array([0.3,0.5,1.0,1.5,2.0,2.5])
    jumpv      = np.array([500,300,150,100,75,60,50])
    ref_factor = 5
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = 2*np.array([0.1])
        vdt   = np.array([0.1])
        jumpv = np.array([1500])

if(ptype==6):
    
    dx         = 5
    dz         = 5
    vmax       = 4.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = 4*np.array([0.06,0.1,0.2,0.3,0.4,0.5])
    vdt        = np.array([0.3,0.5,1.0,1.5,2.0,2.5])
    jumpv      = np.array([500,300,150,100,75,60,50])
    ref_factor = 10
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = 4*np.array([0.2])
        vdt   = np.array([0.1])
        jumpv = np.array([1500])
        
if(ptype==7):
    
    dx         = 15
    dz         = 15
    vmax       = 3.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = np.array([0.1,0.2,0.3,0.4,0.5,0.6])
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    jumpv      = np.array([300,150,100,75,60,50,40])
    ref_factor = 5
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = np.array([0.1])
        vdt   = np.array([0.1])
        jumpv = np.array([1500])

if(ptype==8):
    
    dx         = 10
    dz         = 10
    vmax       = 3.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = np.array([0.15,0.3,0.45,0.6,0.75,0.9])
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    jumpv      = np.array([360,180,120,90,72,60])
    ref_factor = 5
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = np.array([0.15])
        vdt   = np.array([0.1])
        jumpv = np.array([1800])

if(ptype==9):
    
    dx         = 10
    dz         = 10
    vmax       = 8.0
    t0         = 0.0
    tn         = 3000.0
    CFLv       = np.array([0.4,0.8,1.2,1.6,2.0,2.4])
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    jumpv      = np.array([300,150,100,75,60,50,40])
    ref_factor = 5
    
    if(ref==1): 
        
        dx    = dx/ref_factor
        dz    = dz/ref_factor
        CFLv  = np.array([0.4])
        vdt   = np.array([0.1])
        jumpv = np.array([1500])
#==============================================================================

#==============================================================================
# Cacl Delta t
#==============================================================================
ntimes = vdt.shape[0]

for i in range(0,ntimes):
    
    dtmax  = np.round(min(dx,dz)*CFLv[i]/vmax,8)
    ntmax  = int((tn-t0)/dtmax)
    dt0    = (tn-t0)/(ntmax)
    nplot  = mt.ceil(ntmax/jumpv[i]) + 1
    ntjump = jumpv[i]*dt0 
    print(vdt[i],dtmax,ntmax,dt0,nplot,ntjump)
#==============================================================================