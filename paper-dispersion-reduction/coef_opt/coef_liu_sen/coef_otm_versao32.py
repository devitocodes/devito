#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as np
from   scipy.integrate         import quad
#==============================================================================

#==============================================================================
# Funcoes Auxiliares
#==============================================================================

#==============================================================================
def fsis1(i,j,rval,tetaval):

    a = j**(2*i)*(np.cos(tetaval)**(2*i) + np.sin(tetaval)**(2*i))
   
    return a
#==============================================================================

#==============================================================================
def fsis2(i,rval):
 
    a = rval**(2*i-2)
    
    if((rval==0)and(i>1)): a = 0

    return a
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(nordem,rval,tetaval):

    nordersis = int(nordem/2)    

    Asis  = np.zeros((nordersis,nordersis))
    bsis  = np.zeros((nordersis,1))
    csis  = np.zeros((nordersis,1))
    coef  = np.zeros(nordersis+1)
    vcoef = np.zeros(nordem+1)
        
    for i in range(0,nordersis):
  
        bsis[i,0] = fsis2(i+1,rval)
    
        for j in range(0,nordersis):
    
            Asis[i,j] = fsis1(i+1,j+1,rval,tetaval)

    csis = np.linalg.solve(Asis,bsis)
    
    sum_coef = 0
    
    for i in range(0,csis.shape[0]):
        
        sum_coef = sum_coef + ((-1)**(i) + 1)*csis[i,0]

    stab = 1/np.sqrt(sum_coef)
    
    if(rval>stab): print('Infringindo a condição de estabilidade para o método! Cuidado!')

    for i in range(0,nordersis):
        coef[i+1] = csis[i]

    for i in range(1,nordersis+1):
        coef[0] = coef[0] - 2*coef[i]
 
    nmeio = int(nordem/2)   
 
    vcoef[nmeio] = coef[0]   
 
    for i in range(0,nmeio):
        vcoef[i]          = coef[nmeio-i] 
        vcoef[nordem-i]   = coef[nmeio-i]  

    return vcoef
#==============================================================================