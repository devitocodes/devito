#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as np
from   scipy.integrate         import nquad
#==============================================================================

#==============================================================================
# Funcoes Auxiliares
#==============================================================================

#==============================================================================
def fbetasis(beta,teta,m,r):

    a = (r**2)*(2-np.cos(m*beta*np.cos(teta))-np.cos(m*beta*np.sin(teta)))

    b = (1-np.cos(r*beta))

    c = a/b
    
    return c
#==============================================================================

#==============================================================================
def fphisis(beta,teta,m1,m2,r):
 
    a1 = (r**2)*(2-np.cos(m1*beta*np.cos(teta))-np.cos(m1*beta*np.sin(teta)))

    b1 = (1-np.cos(r*beta))

    c1 = a1/b1
    
    a2 = (r**2)*(2-np.cos(m2*beta*np.cos(teta))-np.cos(m2*beta*np.sin(teta)))

    b2 = (1-np.cos(r*beta))

    c2 = a2/b2
    
    c = c1*c2 
    
    return c
#==============================================================================

#==============================================================================
def ferro(bint,tetaint,coef,r):
 
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
            
            a = np.cos(i*beta*np.cos(tetaint))
            
            b = np.cos(i*beta*np.sin(tetaint))
            
            sumv = sumv + coef[i]*(2-a-b)
        
        c = ((r**2)/(2))*sumv
    
        c = ((2)/(r*beta))*np.arcsin(np.sqrt(c))-1
    
        if(np.abs(c)>=abs(erro)): erro = abs(c) 
    
    return erro
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(nordem,bint,tetaint,rval):

    Asis = np.zeros((nordem,nordem))
    bsis = np.zeros((nordem,1))
    csis = np.zeros((nordem,1))
    coef = np.zeros(nordem+1)
    vcoef = np.zeros(2*nordem+1)

    for i in range(0,nordem):
  
        res, err = nquad(fbetasis,[[0,bint],[0,tetaint]],args=(i+1,rval,))
        bsis[i,0] = res
    
        for j in range(0,nordem):
    
            res, err = nquad(fphisis,[[0,bint],[0,tetaint]],args=(i+1,j+1,rval,))
            Asis[i,j] = res

    csis = np.linalg.solve(Asis,bsis)

    sum_coef = 0
    
    for i in range(0,csis.shape[0]):
        
        sum_coef = sum_coef + ((-1)**(i) + 1)*csis[i,0]

    stab = 1/np.sqrt(0.5*sum_coef)

    if(rval>stab): print('Infringindo a condição de estabilidade para o método! Cuidado!')

    for i in range(0,nordem):
        coef[i+1] = csis[i]

    for i in range(1,nordem+1):
        coef[0] = coef[0] - 2*coef[i]

    erro = ferro(bint,tetaint,coef,rval)
    
    for i in range(0,nordem+1):
        vcoef[i]          = coef[nordem-i] 
        vcoef[2*nordem-i] = coef[nordem-i] 
    
    return vcoef
#==============================================================================