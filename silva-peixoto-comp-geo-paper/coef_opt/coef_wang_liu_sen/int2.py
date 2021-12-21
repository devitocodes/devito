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
def pontopeso2(npi):
    
    pontos = np.zeros(npi)
    pesos  = np.zeros(npi)
    m      = int((npi+1)*0.5)

    for i in range(0,m):
        
        cte             = ((2*(i+1)-1)*np.pi)/(2*npi) 
        xtemp           = -np.cos(cte)
        pontos[i]       =  xtemp
        pontos[npi-1-i] = -pontos[i]
        
        ptemp            = np.pi/npi
        pesos[i]         = ptemp
        pesos[npi-1-i]   = pesos[i]
        
    return pontos,pesos
#==============================================================================

#==============================================================================
def pontopeso3(npi):
    
    pontos = np.zeros(npi)
    pesos  = np.zeros(npi)
    m      = int((npi+1)*0.5)

    for i in range(0,m):
        
        cte             = ((i+1)*np.pi)/(npi+1)
        xtemp           =  np.cos(cte)
        pontos[i]       = -xtemp
        pontos[npi-1-i] = -pontos[i]
        
        cte1             = np.pi/(npi+1)
        cte2             = ((i+1)*np.pi)/(npi+1)
        ptemp            = cte1*np.sin(cte2)**2
        pesos[i]         = ptemp
        pesos[npi-1-i]   = pesos[i]
        
    return pontos,pesos
#==============================================================================

#==============================================================================
def int2dt1(x1,x2,pontosx,pesosx,f):

    S    = 0.0
    npix = pesosx.shape[0]    
    e1   = (x2-x1)*0.5
    e2   = (x2+x1)*0.5

    for i in range(0,npix):
        
        Lx = e1*pontosx[i]+e2

        S = S + pesosx[i]*f(Lx)

    int2dvalue = e1*S

    return int2dvalue
#==============================================================================

#==============================================================================
def int2dt2(x1,x2,pontosx,pesosx,f):

    S    = 0.0
    npix = pesosx.shape[0]
    e1   = (x2-x1)*0.5
    e2   = (x2+x1)*0.5

    for i in range(0,npix):
        
        Lx = e1*pontosx[i]+e2

        S = S + pesosx[i]*f(Lx)

    int2dvalue = S

    return int2dvalue
#==============================================================================

#==============================================================================
def int2dt3(x1,x2,pontosx,pesosx,f):

    S    = 0.0
    npix = pesosx.shape[0]
    e1   = (x2-x1)*0.5
    e2   = (x2+x1)*0.5

    for i in range(0,npix):
        
        Lx = e1*pontosx[i]+e2

        S = S + pesosx[i]*f(Lx)

    int2dvalue = (e1**2)*S

    return int2dvalue
#==============================================================================

#==============================================================================
def fteste1(x):
    
    fvalue = x*np.sin(x)
    
    return fvalue
#==============================================================================

#==============================================================================
def fteste2(x):
    
    fvalue = (x**(7/2))/(np.sqrt((1-x)*x))
    
    return fvalue
#==============================================================================

#==============================================================================
def fteste3(x):
    
    fvalue = np.sqrt((1-x)*x)*x**(7/2)
    
    return fvalue
#==============================================================================

#==============================================================================
# Teste de Integração
#==============================================================================
x1     =  0
x2     =  2*np.pi
npix   = 100
pontosx1,pesosx1 = pontopeso1(npix)
int2d1           = int2dt1(x1,x2,pontosx1,pesosx1,fteste1)
vexato1          = -6.283185307179586
erro1            = np.abs(vexato1-int2d1)

x1     =  0
x2     =  1
npix   = 100
pontosx2,pesosx2 = pontopeso2(npix)
int2d2           = int2dt2(x1,x2,pontosx2,pesosx2,fteste2)
vexato2          = 0.914285714285714
erro2            = np.abs(vexato2-int2d2)

x1     =  0
x2     =  1
npix   = 100
pontosx3,pesosx3 = pontopeso3(npix)
int2d3           = int2dt3(x1,x2,pontosx3,pesosx3,fteste3)
vexato3          = 0.073881673881674
erro3            = np.abs(vexato3-int2d3)

print('')
print('==============================================================')
print('Tipo de Quadratura: Gauss-Legendre')
print('Pontos de Integração Direção X: ',npix)
print('O valor da Integral eh: ',int2d1)
print('O valor do erro eh: %.2e'%erro1)
print('==============================================================')
print('')

print('')
print('==============================================================')
print('Tipo de Quadratura: Gauss-Chebyshev 1ª Especie')
print('Pontos de Integração Direção X: ',npix)
print('O valor da Integral eh: ',int2d2)
print('O valor do erro eh: %.2e'%erro2)
print('==============================================================')
print('')

print('')
print('==============================================================')
print('Tipo de Quadratura: Gauss-Chebyshev 2ª Especie')
print('Pontos de Integração Direção X: ',npix)
print('O valor da Integral eh: ',int2d3)
print('O valor do erro eh: %.2e'%erro3)
print('==============================================================')
print('')
#==============================================================================

#==============================================================================
plot.close("all") # fecha todas as janelas
#==============================================================================