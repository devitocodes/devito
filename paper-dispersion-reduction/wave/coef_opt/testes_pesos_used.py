#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as     np
import sys
#==============================================================================

#==============================================================================
# Yang Liu 2012 Coefficients
#==============================================================================
sys.path.insert(0, './basic_weights')
import basic_weights            as     bw
#==============================================================================

#==============================================================================
# Yang Liu 2012 Coefficients
#==============================================================================
sys.path.insert(0, './coef_yang_liu')
import coef_otm_versao14        as     cotm14
#==============================================================================

#==============================================================================
# Liu and Sen 2009 Coefficients
#==============================================================================
sys.path.insert(0, './coef_liu_sen')
import coef_otm_versao32        as     cotm32
#==============================================================================

#==============================================================================
# Wang, Liu and Sen - Cross/Rombus Stencils - 2016
#==============================================================================
sys.path.insert(0, './coef_wang_liu_sen')
import coef_otm_versao71        as     cotm71
import coef_otm_versao72        as     cotm72
#==============================================================================

#==============================================================================
# Pratical Inputs
#==============================================================================
ptype  = 4
dtype  = 1
nordem = 2
mvalue = int(nordem/2)
nvalue = int(mvalue/2) + 1
nround = 2

if(ptype==1):
    
    dx         = 20
    dz         = 20
    vmax       = 4.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    rval       = (vdt[dtype]*vmax)/min(dx,dz)
    
if(ptype==2):
    
    dx         = 10
    dz         = 10
    vmax       = 4.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    rval       = (vdt[dtype]*vmax)/min(dx,dz)
    
if(ptype==3):
    
    dx         = 20
    dz         = 20
    vmax       = 5.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    rval       = (vdt[dtype]*vmax)/min(dx,dz)
            
if(ptype==4):
    
    dx         = 32 # So is Original
    dz         = 32
    vmax       = 8.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    rval       = 0#(vdt[dtype]*vmax)/min(dx,dz)

print('')
#==============================================================================

#==============================================================================
# Basic Weights
#==============================================================================
T0      = bw.calccoef(nordem)
print('FD Scheme = Classical - Order =  %d'%(nordem))
print(np.round(T0,nround))
print('')
#==============================================================================

#==============================================================================
# T4 Type - Coef Yang Liu - A nearly globally optimal FD scheme based on 2D
# time-space-domain dispersion relation
#==============================================================================
#bint    = 2.74
#tetaint = np.pi/4
#T4      = cotm14.calccoef(mvalue,bint,tetaint,rval)
#print('FD Scheme = Cross2009_TE - Order =  %d'%(nordem))
#print(np.round(T4,nround))
#print('')
#==============================================================================

#==============================================================================
# T7 Type - Coef Liu and Sen - Time–space domain high-order FD method for 2D 
# acoustic  wave equations
#==============================================================================
tetaval   = np.pi/8
T7        = cotm32.calccoef(nordem,rval,tetaval)
print('FD Scheme = Cross2013_LS - Order =  %d'%(nordem))
print(np.round(T7,nround))
print('')
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
T121  = cotm71.calccoef(mvalue,nvalue,rval,rval)  
print('FD Scheme = Cross_Rhombhus_2016_TE - Order =  %d - M = %d - N = %d'%(2*mvalue,mvalue,nvalue))
print(np.round(T121,nround))
print('')
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
#T122 = cotm72.calccoef(mvalue,nvalue,rval,rval)
#print('FD Scheme = Cross_Rhombhus_2016_LS - Order =  %d - M = %d - N = %d'%(2*mvalue,mvalue,nvalue))
#print(np.round(T122,nround))
#print('')
#==============================================================================