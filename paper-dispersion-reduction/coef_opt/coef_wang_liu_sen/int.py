#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================
# Módulos Importados do Python / Devito / Examples
#==============================================================================
#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as np
import matplotlib.pyplot       as plot
import math                    as mt
import sys
import time                    as tm
import sympy                   as sym
from   numpy                   import linalg as la
#==============================================================================

#==============================================================================
def pontopeso1(npi):
    
    pontos = np.zeros(npi)
    pesos  = np.zeros(npi)
    
    if(npi==1):
    
        pontos[0] = 0.0
        pesos[0]  = 2.0

    if(npi==2):
     
        pontos[0] = -0.577350269189626
        pontos[1] = -pontos[0]
        
        pesos[0]  =  1.0
        pesos[1]  =  pesos[0]
    
    if(npi==3):
     
        pontos[0]  = -0.774596669241483
        pontos[1]  =  0.0
        pontos[2]  = -pontos[0]
    
        pesos[0]   = 0.555555555555556
        pesos[1]   = 0.888888888888889
        pesos[2]   = pesos[0]
        
    if(npi==4):
      
        pontos[0]  = -0.861136311594053
        pontos[1]  = -0.339981043584856
        pontos[2]  = -pontos[1]
        pontos[3]  = -pontos[0]
    
        pesos[0]   = 0.347854845137454
        pesos[1]   = 0.652145154862546
        pesos[2]   = pesos[1]
        pesos[3]   = pesos[0]
 
    if(npi==5):
    
        pontos[0]  = -0.906179845938664
        pontos[1]  = -0.538469310105683
        pontos[2]  =  0.0
        pontos[3]  = -pontos[1]
        pontos[4]  = -pontos[0]
    
        pesos[0]   = 0.236926885056189
        pesos[1]   = 0.478628670499366
        pesos[2]   = 0.568888888888889
        pesos[3]   = pesos[1]
        pesos[4]   = pesos[0]
        
    if(npi==6):
    
        pontos[0]  = -0.9324695142
        pontos[1]  = -0.6612093864
        pontos[2]  = -0.2386191860
        pontos[3]  = -pontos[2]
        pontos[4]  = -pontos[1]    
        pontos[5]  = -pontos[0]    
    
        pesos[0]   = 0.1713244923
        pesos[1]   = 0.3607615730
        pesos[2]   = 0.4679139345
        pesos[3]   = pesos[2]
        pesos[4]   = pesos[1]
        pesos[5]   = pesos[0]
    
    if(npi==7):
    
        pontos[0]  = -0.9491079123
        pontos[1]  = -0.7415311855
        pontos[2]  = -0.4058451513
        pontos[3]  =  0.0
        pontos[4]  = -pontos[2]
        pontos[5]  = -pontos[1]
        pontos[6]  = -pontos[0]
        
        pesos[0]   = 0.1294849661
        pesos[1]   = 0.2797053914
        pesos[2]   = 0.3818300505
        pesos[3]   = 0.4179591836
        pesos[4]   = pesos[2]
        pesos[5]   = pesos[1]
        pesos[6]   = pesos[0]

    if(npi==8):
    
        pontos[0]  = -0.9602898564
        pontos[1]  = -0.7966664774
        pontos[2]  = -0.5255324099
        pontos[3]  = -0.1834346424
        pontos[4]  = -pontos[3]
        pontos[5]  = -pontos[2]
        pontos[6]  = -pontos[1]
        pontos[7]  = -pontos[0]
          
        pesos[0]   = 0.1012285362
        pesos[1]   = 0.2223810344
        pesos[2]   = 0.3137066458
        pesos[3]   = 0.3626837833
        pesos[4]   = pesos[3]
        pesos[5]   = pesos[2]
        pesos[6]   = pesos[1]
        pesos[7]   = pesos[0]

    return pontos,pesos
#==============================================================================

#==============================================================================
def pontopeso2(npi):
    
    pontos = np.zeros(npi)
    pesos  = np.zeros(npi)
    m      = int((npi+1)*0.5)

    for i in range(0,m):
        
        cte1 = np.pi*(i+1-0.25)/(npi+0.5)
        
        x = np.cos(cte1)
        
        while(True):
        
            p1 = 1
            p2 = 0
            
            for j in range(0,npi):
                
                p3 = p2
                p2 = p1
                p1 = ((2*j+1)*x*p2-j*p3)/(j+1)
            
            pp = npi*(x*p1-p2)/(x**2-1)
            
            x1 = x
            x  = x1 - (p1/pp)
            
            if(abs(x-x1)<1e-15):
            
                break
        
        pontos[i]       = -x
        pontos[npi-1-i] = -pontos[i]
        
        pesos[i]         = 2/((1-x**2)*pp**2)
        pesos[npi-1-i]   = pesos[i]
        
    return pontos,pesos
#==============================================================================

#==============================================================================
def int2d(x1,x2,y1,y2,pontosx,pesosx,pontosy,pesosy,f):

    S    = 0.0
    npix = pesosx.shape[0]
    npiy = pesosy.shape[0]

    for i in range(0,npix):
        
        Lx = (x2-x1)*0.5*pontosx[i]+(x2+x1)*0.5

        for j in range(0,npiy):

            Ly = (y2-y1)*0.5*pontosy[j]+(y2+y1)*0.5;

            S = S + pesosx[i]*pesosy[j]*f(Lx,Ly);

    int2dvalue = (x2-x1)*0.5*(y2-y1)*0.5*S

    return int2dvalue
#==============================================================================

#==============================================================================
def romberg1d(f,a,b,n):

    r = np.zeros((n+1,n+1))

    h = b - a

    r[0,0] = 0.5*h*(f(a)+f(b))

    cont1 = 1

    for i in range(1,n+1):

        h = 0.5*h

        sumf = 0.0
        
        cont1 = 2*cont1

        for k in range(1,cont1,2):

            xloc = a+k*h            

            sumf = sumf + f(xloc)

        r[i,0] = 0.5*r[i-1,0] + sumf*h

        cont2 = 1

        for j in range(1,i+1):
            
            cont2 = 4*cont2
            
            r[i,j] = r[i,j-1] + (r[i,j-1]-r[i-1,j-1])/(cont2-1)

    intvalue = r[-1,-1]

    return intvalue
#==============================================================================

#==============================================================================
def fteste1(x,y):
    
    fvalue = (np.sin(x+2)-np.exp(-x**2))/(x**2+np.log(x+2))
    
    return fvalue
#==============================================================================

#==============================================================================
def fteste2(x):
    
    fvalue = (np.sin(x+2)-np.exp(-x**2))/(x**2+np.log(x+2))
    
    return fvalue
#==============================================================================

#==============================================================================
# Teste de Integração
#==============================================================================
x1   = -1
x2   =  1
y1   =  0
y2   =  1

for i in range(1,9):

    npix = i
    npiy = i    

    pontosx1,pesosx1 = pontopeso1(npix)
    pontosy1,pesosy1 = pontopeso1(npiy)
    pontosx2,pesosx2 = pontopeso2(npix)
    pontosy2,pesosy2 = pontopeso2(npiy)
    int2d1           = int2d(x1,x2,y1,y2,pontosx1,pesosx1,pontosy1,pesosy1,fteste1)
    int2d2           = int2d(x1,x2,y1,y2,pontosx2,pesosx2,pontosy2,pesosy2,fteste1)
    
    dif         = np.abs(int2d1-int2d2)
    normpesosx  = np.linalg.norm(pesosx1-pesosx2,2)
    normpesosy  = np.linalg.norm(pesosy1-pesosy2,2)
    normpontosx = np.linalg.norm(pontosx1-pontosx2,2)
    normpontosy = np.linalg.norm(pontosy1-pontosy2,2)
    
    print('')
    print('==============================================================')
    print('Pontos de Integração Direção X: ',npix)
    print('Pontos de Integração Direção Y: ',npiy)
    print('O valor da Integral com pontopesos1 eh: ',int2d1)
    print('O valor da Integral com pontopesos2 eh: ',int2d2)
    print('A diferença nas integrações eh: %.2e'%dif)
    print('A norma da diferença dos pesos na direção X eh: %.2e'%normpesosx)
    print('A norma da diferença dos pesos na direção Y eh: %.2e'%normpesosy)
    print('A norma da diferença dos pontos na direção X eh: %.2e'%normpontosx)
    print('A norma da diferença dos pontos na direção Y eh: %.2e'%normpontosy)
    print('==============================================================')
    print('')
#==============================================================================

r1 = romberg1d(fteste2,x1,x2,10)
print(r1)

#==============================================================================
plot.close("all") # fecha todas as janelas
#==============================================================================