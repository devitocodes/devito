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
def fpoints(nordem,x0,y0,dx,dy):

    npoints = 2*nordem + 1    
    
    vpoints = np.zeros((npoints,2))    
    
    conti = 0
    contf = 0
    
    vpoints[0,0] = x0
    vpoints[0,1] = y0
    
    nmove = int(nordem/2)
    
    conti = 1
    contf = nmove + 1
     
    for i in range(conti,contf):
        vpoints[i,0] = - (nmove-i+1)*dx - x0
        vpoints[i,1] = y0
    
    conti = contf
    contf = contf + nmove
    
    for i in range(conti,contf):
        vpoints[i,0] = (i-conti+1)*dx - x0
        vpoints[i,1] = y0
  
    conti = contf
    contf = contf + nmove
    
    for i in range(conti,contf):
        vpoints[i,0] = x0
        vpoints[i,1] = -(nmove-i+conti)*dy - y0
        
    conti = contf
    contf = contf + nmove
    
    for i in range(conti,contf):
        vpoints[i,0] = x0
        vpoints[i,1] = (i-conti+1)*dy - y0
          
    return vpoints
#==============================================================================

#==============================================================================
def fphisis1(tetaval,kval,vpoints,m1,m2,dscale):
 
    dx1 = vpoints[m1,0]
    dy1 = vpoints[m1,1]
         
    l1  = dx1*np.cos(tetaval) + dy1*np.sin(tetaval)
    a1  = np.cos(dscale*kval*l1)
    b1  = np.sin(dscale*kval*l1)
 
    dx2 = vpoints[m2,0]
    dy2 = vpoints[m2,1]
       
    l2  = dx2*np.cos(tetaval) + dy2*np.sin(tetaval)
    a2  = np.cos(dscale*kval*l2)
    b2  = np.sin(dscale*kval*l2)
    
    c   = a1*a2 + b1*b2

    return c
#==============================================================================

#==============================================================================
def fphisis2(tetaval,kval,vpoints,rval,m1,dscale):
 
    dx1 = vpoints[m1,0]
    dy1 = vpoints[m1,1]

    rval = rval/dscale
     
    l1  = dx1*np.cos(tetaval) + dy1*np.sin(tetaval)
    a1  = np.cos(dscale*kval*l1)

    b1  = np.cos(dscale*kval*rval) - 1 
        
    c   = (2/(rval**2))*a1*b1

    return c
#==============================================================================

#==============================================================================
def ferro(tetaval,kval,coef,vpoints,rval,dscale):

    rval    = rval/dscale
    npoints = coef.shape[0]
    erro    = 0
    soma1   = 0   
    soma2   = 0   
 
    for i in range(0,npoints):   
 
        dx = vpoints[i,0]
        dy = vpoints[i,1]
        
        l  = dx*np.cos(tetaval) + dy*np.sin(tetaval)
        a  = np.cos(dscale*kval*l)
        b  = np.sin(dscale*kval*l)
    
        soma1 = soma1 + a*coef[i,0]
        soma2 = soma2 + b*coef[i,0]
        
    c  = (2/(rval**2))*(np.cos(dscale*kval*rval) - 1) 
    
    soma1 = soma1 - c
    
    erro = soma1**2 + soma2**2
    
    return erro
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(nordem,K,x0,y0,dx,dy,rval):

    dscale = max(dx,dy)    

    #K = K*dscale
    
    vpoints = fpoints(nordem,x0,y0,dx,dy)

    vpoints = vpoints/dscale
    
    npoints = 2*nordem + 1

    Asis    = np.zeros((npoints,npoints))
    bsis    = np.zeros((npoints,1))
    csis    = np.zeros((npoints,1))
    coefxx  = np.zeros(nordem+1)
    coefyy  = np.zeros(nordem+1)

    tetaint = np.pi

    for i in range(0,npoints):
  
        res, err = nquad(fphisis2,[[0,tetaint],[0,K]],args=(vpoints,rval,i,dscale,))
        bsis[i,0] = res
    
        for j in range(0,npoints):
    
            res, err = nquad(fphisis1,[[0,tetaint],[0,K]],args=(vpoints,i,j,dscale,))
            Asis[i,j] = res

    csis = np.linalg.solve(Asis,bsis)
    
    res, errom = nquad(ferro,[[0,2*tetaint],[0,K]],args=(csis,vpoints,rval,dscale,))

    nmedio = int(nordem/2)
    
    #csis = csis/dscale**2
    
    coefxx[nmedio] = csis[0,0]
    coefyy[nmedio] = csis[0,0]
    
    for i in range(1,nmedio+1):
        coefxx[i-1]      = csis[i,0]
        coefxx[nmedio+i] = csis[nmedio+i,0]
        coefyy[i-1]      = csis[nordem+i,0]
        coefyy[nmedio+i] = csis[nordem+nmedio+i,0]
        
    sum1   = 4
    sum2x  = 0
    sum2y  = 0
    
    for i in range(0,nordem+1):
        
        sum2x = sum2x + np.abs(coefxx[i])
        sum2y = sum2y + np.abs(coefyy[i])
    
    paramx = np.sqrt(sum1/sum2x)
    paramy = np.sqrt(sum1/sum2y)
        
    return coefxx, coefyy
#==============================================================================