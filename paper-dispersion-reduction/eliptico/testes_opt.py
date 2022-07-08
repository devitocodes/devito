#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================

#==============================================================================
# Módulos Importados do Python / Devito / Examples
#==============================================================================

#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
#==============================================================================

#==============================================================================
class teste1:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    teste_type       = 1
    dtype            = np.float64
    nptx             = 101
    nptz             = 101
    x0               = 0.0
    x1               = 1.0 
    compx            = x1-x0
    z0               = 0.0
    z1               = 1.0
    compz            = z1-z0   
    hx               = (x1-x0)/(nptx-1)           
    hz               = (z1-z0)/(nptz-1)               
    X                = np.linspace(x0,x1,nptx)    
    Z                = np.linspace(z0,z1,nptz)
    sou              = 2
    wauthor          = 3                        
    wtype            = 1
    nvalue           = np.random.randint(low=1,high=int(sou/2)+1,size=1)[0]
    exttrapmax       = int(0.5*(int(sou/2)**2+int(sou/2)))
    exttrap          = np.random.randint(low=0,high=exttrapmax,size=1)[0]
    nptx_ext         = nptx + 2*sou
    nptz_ext         = nptz + 2*sou
    Xext             = np.linspace(x0-sou*hx,x1+sou*hx,nptx_ext)
    Zext             = np.linspace(z0-sou*hz,z1+sou*hz,nptz_ext)
    normtype         = 2
#==============================================================================

#==============================================================================
class teste2:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    teste_type       = 2
    dtype            = np.float64
    nptx             = 101
    nptz             = 101
    x0               = 0.0
    x1               = 1.0 
    compx            = x1-x0
    z0               = 0.0
    z1               = 1.0
    compz            = z1-z0   
    hx               = (x1-x0)/(nptx-1)           
    hz               = (z1-z0)/(nptz-1)               
    X                = np.linspace(x0,x1,nptx)    
    Z                = np.linspace(z0,z1,nptz)
    sou              = 2                               
    wauthor          = 3                                
    wtype            = 1
    nvalue           = np.random.randint(low=1,high=int(sou/2)+1,size=1)[0]
    exttrapmax       = int(0.5*(int(sou/2)**2+int(sou/2)))+1
    exttrap          = np.random.randint(low=0,high=exttrapmax,size=1)[0]
    nptx_ext         = nptx + 2*sou
    nptz_ext         = nptz + 2*sou
    Xext             = np.linspace(x0-sou*hx,x1+sou*hx,nptx_ext)
    Zext             = np.linspace(z0-sou*hz,z1+sou*hz,nptz_ext)
    normtype         = 2
#==============================================================================

#==============================================================================
class teste3:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    teste_type       = 3
    dtype            = np.float64
    nptx             = 101
    nptz             = 101
    x0               = 0.0
    x1               = 1.0 
    compx            = x1-x0
    z0               = 0.0
    z1               = 1.0
    compz            = z1-z0   
    hx               = (x1-x0)/(nptx-1)           
    hz               = (z1-z0)/(nptz-1)               
    X                = np.linspace(x0,x1,nptx)    
    Z                = np.linspace(z0,z1,nptz)
    sou              = 2                               
    wauthor          = 3                                
    wtype            = 1
    nvalue           = np.random.randint(low=1,high=int(sou/2)+1,size=1)[0]
    exttrapmax       = int(0.5*(int(sou/2)**2+int(sou/2)))+1
    exttrap          = np.random.randint(low=0,high=exttrapmax,size=1)[0]
    nptx_ext         = nptx + 2*sou
    nptz_ext         = nptz + 2*sou
    Xext             = np.linspace(x0-sou*hx,x1+sou*hx,nptx_ext)
    Zext             = np.linspace(z0-sou*hz,z1+sou*hz,nptz_ext)
    normtype         = 2
#==============================================================================

#==============================================================================
class teste4:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    teste_type       = 4
    dtype            = np.float64
    nptx             = 101
    nptz             = 101
    x0               = 0.0
    x1               = 1.0 
    compx            = x1-x0
    z0               = 0.0
    z1               = 1.0
    compz            = z1-z0   
    hx               = (x1-x0)/(nptx-1)           
    hz               = (z1-z0)/(nptz-1)               
    X                = np.linspace(x0,x1,nptx)    
    Z                = np.linspace(z0,z1,nptz)
    sou              = 2                               
    wauthor          = 3                                
    wtype            = 1
    nvalue           = np.random.randint(low=1,high=int(sou/2)+1,size=1)[0]
    exttrapmax       = int(0.5*(int(sou/2)**2+int(sou/2)))+1
    exttrap          = np.random.randint(low=0,high=exttrapmax,size=1)[0]
    nptx_ext         = nptx + 2*sou
    nptz_ext         = nptz + 2*sou
    Xext             = np.linspace(x0-sou*hx,x1+sou*hx,nptx_ext)
    Zext             = np.linspace(z0-sou*hz,z1+sou*hz,nptz_ext)
    normtype         = 2
#==============================================================================

#==============================================================================
class teste5:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    teste_type       = 5
    dtype            = np.float64
    nptx             = 101
    nptz             = 101
    x0               = -0.5
    x1               = 1.0 
    compx            = x1-x0
    z0               = -0.5
    z1               = 1.0
    compz            = z1-z0   
    hx               = (x1-x0)/(nptx-1)           
    hz               = (z1-z0)/(nptz-1)               
    X                = np.linspace(x0,x1,nptx)    
    Z                = np.linspace(z0,z1,nptz)
    sou              = 2                                
    wauthor          = 3                                
    wtype            = 1
    nvalue           = np.random.randint(low=1,high=int(sou/2)+1,size=1)[0]
    exttrapmax       = int(0.5*(int(sou/2)**2+int(sou/2)))+1
    exttrap          = np.random.randint(low=0,high=exttrapmax,size=1)[0]
    nptx_ext         = nptx + 2*sou
    nptz_ext         = nptz + 2*sou
    Xext             = np.linspace(x0-sou*hx,x1+sou*hx,nptx_ext)
    Zext             = np.linspace(z0-sou*hz,z1+sou*hz,nptz_ext)
    normtype         = 2
#==============================================================================

#==============================================================================
class teste6:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    teste_type       = 6
    dtype            = np.float64
    nptx             = 101
    nptz             = 101
    x0               = -0.5
    x1               = 1.0 
    compx            = x1-x0
    z0               = -0.5
    z1               = 1.0
    compz            = z1-z0   
    hx               = (x1-x0)/(nptx-1)           
    hz               = (z1-z0)/(nptz-1)               
    X                = np.linspace(x0,x1,nptx)    
    Z                = np.linspace(z0,z1,nptz)
    sou              = 2                                
    wauthor          = 3                                
    wtype            = 1
    nvalue           = np.random.randint(low=1,high=int(sou/2)+1,size=1)[0]
    exttrapmax       = int(0.5*(int(sou/2)**2+int(sou/2)))+1
    exttrap          = np.random.randint(low=0,high=exttrapmax,size=1)[0]
    nptx_ext         = nptx + 2*sou
    nptz_ext         = nptz + 2*sou
    Xext             = np.linspace(x0-sou*hx,x1+sou*hx,nptx_ext)
    Zext             = np.linspace(z0-sou*hz,z1+sou*hz,nptz_ext)
    normtype         = 2
#==============================================================================

#==============================================================================
class teste7:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    teste_type       = 7
    dtype            = np.float64
    nptx             = 101
    nptz             = 101
    x0               = 0.0
    x1               = 1.0 
    compx            = x1-x0
    z0               = 0.0
    z1               = 1.0
    compz            = z1-z0   
    hx               = (x1-x0)/(nptx-1)           
    hz               = (z1-z0)/(nptz-1)               
    X                = np.linspace(x0,x1,nptx)    
    Z                = np.linspace(z0,z1,nptz)
    sou              = 4                               
    wauthor          = 3                                
    wtype            = 1
    nvalue           = np.random.randint(low=1,high=int(sou/2)+1,size=1)[0]
    exttrapmax       = int(0.5*(int(sou/2)**2+int(sou/2)))+1
    exttrap          = 3#np.random.randint(low=0,high=exttrapmax,size=1)[0]
    nptx_ext         = nptx + 2*sou
    nptz_ext         = nptz + 2*sou
    Xext             = np.linspace(x0-sou*hx,x1+sou*hx,nptx_ext)
    Zext             = np.linspace(z0-sou*hz,z1+sou*hz,nptz_ext)
    normtype         = 2
#==============================================================================