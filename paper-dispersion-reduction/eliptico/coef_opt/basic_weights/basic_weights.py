#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
from   numpy import linalg as la
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg        import solve
#==============================================================================

#==============================================================================
# Print Configurations
#==============================================================================
np.set_printoptions(formatter={'float': '{: 0.4e}'.format})
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(nordem):
#==============================================================================

#==============================================================================    
    nordersis = int(nordem/2)    

    Asis  = np.zeros((nordersis,nordersis))
    bsis  = np.zeros((nordersis,1))
    csis  = np.zeros((nordersis,1))
    coef  = np.zeros(nordersis+1)
    vcoef = np.zeros(nordem+1)
        
    for j in range(0,nordersis):

        if(j==0): bsis[j] = 1

        for i in range(0,nordersis):
        
            Asis[j,i] = (i+1)**(2*(j+1))
#==============================================================================

#==============================================================================
    ncond = la.cond(Asis,2) 
    print('')
    print('====================================================')
    print('The Condition Number of System Matrix is: %.3e'%ncond)
    print('====================================================')
    print('')
    solver_system = 0
#==============================================================================

#==============================================================================
    if(solver_system==0):
        csis = la.solve(Asis,bsis)
        print('====================================')
        print('System Solver: np.linalg.solve')
        print('====================================')
#==============================================================================

#==============================================================================    
    if(solver_system==1):
        csis = lu_solve((lu_factor(Asis)), bsis)
        print('====================================')
        print('System Solver: lu_solve')
        print('====================================')
#==============================================================================

#==============================================================================
    for i in range(0,nordersis):
        coef[i+1] = csis[i]

    for i in range(1,nordersis+1):
        coef[0] = coef[0] - 2*coef[i]
 
    nmeio = int(nordem/2)   
 
    vcoef[nmeio] = coef[0]   
 
    for i in range(0,nmeio):
        vcoef[i]          = coef[nmeio-i] 
        vcoef[nordem-i]   = coef[nmeio-i]  
    
    mshow = np.zeros((nordem+1,nordem+1))
    nmeio = int(nordem/2)
    
    mshow[nmeio,:]     = vcoef[:]
    mshow[:,nmeio]     = vcoef[:]
    mshow[nmeio,nmeio] = 2*vcoef[nmeio]
    
    return vcoef, mshow
#==============================================================================