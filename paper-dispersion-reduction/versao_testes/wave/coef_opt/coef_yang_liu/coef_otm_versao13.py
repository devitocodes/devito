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
def fbetasis(beta,m,r):

    a = 1.0

    b = (r**2)*(1-np.cos(m*beta))/(1-np.cos(r*beta))

    c = a*b
    
    return c
#==============================================================================

#==============================================================================
def fphisis(beta,m1,m2,r):
 
    a = (r**2)*(1-np.cos(m1*beta))/(1-np.cos(r*beta))

    b = (r**2)*(1-np.cos(m2*beta))/(1-np.cos(r*beta))

    c = a*b

    return c
#==============================================================================

#==============================================================================
def ferro(bint,coef,r):
 
    n    = coef.shape[0] 
 
    erro = 0
    
    nptaval = 1000
    
    eps = 10^(-8)
    
    vbeta   = np.linspace(eps,bint,nptaval) 
     
    for k in range(0,nptaval):   
 
        c    = 0
        
        sumv = 0
        
        beta = vbeta[k]
        
        for i in range(1,n):
                
            sumv = sumv + coef[i]*(1-np.cos(i*beta))
        
        c = ((r**2)/(2))*sumv
    
        c = ((2)/(r*beta))*np.arcsin(np.sqrt(c))-1
    
        if(np.abs(c)>=abs(erro)): erro = abs(c) 
    
    return erro
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(nordem,bint,rval):

    Asis = np.zeros((nordem,nordem))
    bsis = np.zeros((nordem,1))
    csis = np.zeros((nordem,1))
    coef = np.zeros(nordem+1)
    vcoef = np.zeros(2*nordem+1)
    
    for i in range(0,nordem):
  
        res, err = quad(fbetasis,0,bint,args=(i+1,rval,))
        bsis[i,0] = res
    
        for j in range(0,nordem):
    
            res, err = quad(fphisis,0,bint,args=(i+1,j+1,rval,))
            Asis[i,j] = res

    csis = np.linalg.solve(Asis,bsis)

    sum_coef = 0
    
    for i in range(0,csis.shape[0]):
        
        sum_coef = sum_coef + ((-1)**(i) + 1)*csis[i,0]

    stab = 1/np.sqrt(sum_coef)

    if(rval>stab): print('Infringindo a condição de estabilidade para o método! Cuidado!')

    for i in range(0,nordem):
        coef[i+1] = csis[i]
    
    for i in range(1,nordem+1):
        coef[0] = coef[0] - 2*coef[i]

    erro = ferro(bint,coef,rval)

    for i in range(0,nordem+1):
        vcoef[i]          = coef[nordem-i] 
        vcoef[2*nordem-i] = coef[nordem-i] 
        
    return vcoef
#==============================================================================