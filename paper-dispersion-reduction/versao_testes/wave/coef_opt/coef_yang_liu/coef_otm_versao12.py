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
def fbetasis(beta,m):

    a = 1.0

    b = (2 - 2*np.cos(m*beta))/((beta)**2)

    c = a*b
    
    return c
#==============================================================================

#==============================================================================
def fphisis(beta,m1,m2):
 
    a = (2 - 2*np.cos(m1*beta))/((beta)**2)

    b = (2 - 2*np.cos(m2*beta))/((beta)**2)

    c = a*b

    return c
#==============================================================================

#==============================================================================
def fphierro(beta,m):
 
    a = 2 - 2*np.cos(m*beta)

    return a
#==============================================================================

#==============================================================================
def fbetaerro(beta):

    a = (beta)**2

    return a
#==============================================================================

#==============================================================================
def ferro1(beta,coef):
 
    n = coef.shape[0] 
 
    c = 0   
 
    for i in range(0,n):
        
        a = fphierro(beta,i)
        
        c = c + (coef[i]*a)
    
    b = fbetaerro(beta)

    c = c - b

    c = c**2
    
    return c
#==============================================================================

#==============================================================================
def ferro3(bint,coef):
 
    n    = coef.shape[0] 
 
    erro = 0
    
    nptaval = 1000
    
    vbeta   = np.linspace(0,bint,nptaval) 
     
    for k in range(0,nptaval):   
 
        c = 0
        
        beta = vbeta[k]
        
        for i in range(0,n):
        
            a = fphierro(beta,i)
        
            c = c + (coef[i]*a)
    
        b = fbetaerro(beta)
    
        c = c - b
    
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
  
        res, err = quad(fbetasis,0,bint,args=(i+1,))
        bsis[i,0] = res
    
        for j in range(0,nordem):
    
            res, err = quad(fphisis,0,bint,args=(i+1,j+1,))
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

    res, err = quad(ferro1,0,bint,args=(coef,))
    erro1    = res
    erro2    = np.sqrt((1/bint)*res) 
    erro3    = ferro3(bint,coef)

    for i in range(0,nordem+1):
        vcoef[i]          = coef[nordem-i] 
        vcoef[2*nordem-i] = coef[nordem-i] 
    
    return vcoef
#==============================================================================