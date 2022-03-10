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
import math  as mt
#==============================================================================

#==============================================================================
class teste1:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 101     # Número de Pontos Direção X
    npty   = 101     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 1000.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.06,0.1,0.2,0.3,0.4,0.5]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                         # Condição CFL
    jumpv      = np.array([500,300,150,100,75,60])    # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                    # Intervalo de Dados a Serem Salvos
    tou        = 2                                # Time Order Displacement 
    sou        = 2                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 3                                # Boundary Type    
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste1_ref1:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 501     # Número de Pontos Direção X
    npty   = 501     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 1000.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.1])       # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                    # Condição CFL
    jumpv      = np.array([1500])                # Vetor Intervalo de Dados a Serem Salvos 
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 3                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste2:
# Configuração Non-Absorbing para o Teste 3
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 251     # Número de Pontos Direção X
    npty   = 251     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2500.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2500.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1250.    # Posição da Fonte em X
    yposf  = 1100.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.2,0.4,0.6,0.8,1.0,1.2]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                     # Condição CFL
    jumpv      = np.array([300,150,100,75,60,50]) # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                    # Intervalo de Dados a Serem Salvos
    tou        = 2                                # Time Order Displacement 
    sou        = 2                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 1                                # Boundary Type    
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste2_ref1:
# Configuração Non-Absorbing para o Teste 3
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 1251    # Número de Pontos Direção X
    npty   = 1251    # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2500.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2500.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1250.    # Posição da Fonte em X
    yposf  = 1100.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.2])       # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                    # Condição CFL
    jumpv      = np.array([1500])                # Vetor Intervalo de Dados a Serem Salvos 
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 1                               # Boundary Type    
    ftype      = 0                               # Source type                                                     
#==============================================================================

#==============================================================================
class teste3:
# Configuração Non-Absorbing para o Teste 2
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 601     # Número de Pontos Direção X
    npty   = 201     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 12000.  # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     =  4000.  # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 6000.    # Posição da Fonte em X
    yposf  =   20.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  20.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.125,0.25,0.375,0.5,0.625,0.75]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                     # Condição CFL
    jumpv      = np.array([300,150,100,75,60,50]) # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                    # Intervalo de Dados a Serem Salvos
    tou        = 2                                # Time Order Displacement 
    sou        = 2                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 2                                # Boundary Type
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#==============================================================================

#==============================================================================
class teste3_ref1:
# Configuração Non-Absorbing para o Teste 2
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 3001    # Número de Pontos Direção X
    npty   = 1001    # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 12000.  # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     =  4000.  # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 6000.    # Posição da Fonte em X
    yposf  =   20.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  20.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.125])     # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                    # Condição CFL
    jumpv      = np.array([1500])                # Vetor Intervalo de Dados a Serem Salvos 
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#==============================================================================

#==============================================================================
class teste5:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 201     # Número de Pontos Direção X
    npty   = 201     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 1000.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = 2*np.array([0.06,0.1,0.2,0.3,0.4,0.5]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                         # Condição CFL
    jumpv      = np.array([500,300,150,100,75,60])    # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                    # Intervalo de Dados a Serem Salvos
    tou        = 2                                # Time Order Displacement 
    sou        = 2                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 3                                # Boundary Type    
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste5_ref1:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 1001    # Número de Pontos Direção X
    npty   = 1001    # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 1000.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = 2*np.array([0.1])       # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                    # Condição CFL
    jumpv      = np.array([1500])                # Vetor Intervalo de Dados a Serem Salvos 
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 3                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste6:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 401     # Número de Pontos Direção X
    npty   = 401     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 1000.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = 4*np.array([0.06,0.1,0.2,0.3,0.4,0.5]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                         # Condição CFL
    jumpv      = np.array([500,300,150,100,75,60])    # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                    # Intervalo de Dados a Serem Salvos
    tou        = 2                                # Time Order Displacement 
    sou        = 2                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 3                                # Boundary Type    
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste6_ref1:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 2001    # Número de Pontos Direção X
    npty   = 2001    # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 1000.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = 4*np.array([0.1])       # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                    # Condição CFL
    jumpv      = np.array([1500])                # Vetor Intervalo de Dados a Serem Salvos 
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 3                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste7:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 201     # Número de Pontos Direção X
    npty   = 201     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 3000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 3000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.03     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1500.    # Posição da Fonte em X
    yposf  = 1500.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1500                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.1,0.2,0.3,0.4,0.5,0.6])  # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                         # Condição CFL
    jumpv      = np.array([300,150,100,75,60,50])     # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                    # Intervalo de Dados a Serem Salvos
    tou        = 2                                # Time Order Displacement 
    sou        = 2                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 3                                # Boundary Type    
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste7_ref1:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 1001    # Número de Pontos Direção X
    npty   = 1001    # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 3000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 3000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.03     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1500.    # Posição da Fonte em X
    yposf  = 1500.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1500                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.1])       # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                    # Condição CFL
    jumpv      = np.array([1500])                # Vetor Intervalo de Dados a Serem Salvos 
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 3                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste8:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 201     # Número de Pontos Direção X
    npty   = 201     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.025    # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 800.     # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000.                     # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.15,0.3,0.45,0.6,0.75,0.9])  # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                            # Condição CFL
    jumpv      = np.array([360,180,120,90,72,60])        # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                    # Intervalo de Dados a Serem Salvos
    tou        = 2                                # Time Order Displacement 
    sou        = 2                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 1                                # Boundary Type    
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste8_ref1:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 1001    # Número de Pontos Direção X
    npty   = 1001    # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.025    # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 800.     # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000.                     # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.15])       # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                    # Condição CFL
    jumpv      = np.array([1800])                # Vetor Intervalo de Dados a Serem Salvos 
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 1                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste9:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 201     # Número de Pontos Direção X
    npty   = 201     # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.025    # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 800.     # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000.                     # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.4,0.8,1.2,1.6,2.0,2.0])     # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                            # Condição CFL
    jumpv      = np.array([300,150,100,75,60,60])       # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                    # Intervalo de Dados a Serem Salvos
    tou        = 2                                # Time Order Displacement 
    sou        = 2                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 1                                # Boundary Type    
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste9_ref1:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 1001    # Número de Pontos Direção X
    npty   = 1001    # Número de Pontos Direção Y
    x0     =    0.   # Ponto Inicial Direção X
    x1     = 2000.   # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     = 2000.   # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.025    # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 800.     # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000.                     # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.4])       # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                    # Condição CFL
    jumpv      = np.array([1500])                # Vetor Intervalo de Dados a Serem Salvos 
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 1                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================
