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
    tn     = 2000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 1000.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.075,0.15,0.24,0.3,0.375]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jumpv      = np.array([200,100,63,50,40])    # Vetor com Diferentes Valores de Jump
    jump       = jumpv[0]                        # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 2                               # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 1                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste1_ref1:
# Configuração Non-Absorbing para o Teste 1
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
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
    tn     = 2000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 1000.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.15])      # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jump       = 1000                            # Intervalo de Dados a Serem Salvos
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
class teste2:
# Configuração Non-Absorbing para o Teste 2
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
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
    tn     = 4000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 6000.    # Posição da Fonte em X
    yposf  =   20.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  20.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.09,0.225,0.36,0.45,0.54]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jumpv      = np.array([500,200,125,100,84])    # Vetor com Diferentes Valores de Jump
    jump       = jumpv[0]                        # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 2                               # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 1                               # Allow Different Weights      
    wauthor    = 7                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#==============================================================================

#==============================================================================
class teste2_ref1:
# Configuração Non-Absorbing para o Teste 2
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
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
    tn     = 4000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 6000.    # Posição da Fonte em X
    yposf  =   20.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  20.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.225])    # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jump       = 1000                            # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 4                               # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#==============================================================================

#==============================================================================
class teste3:
# Configuração Non-Absorbing para o Teste 3
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
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
    tn     = 2000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02     # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 800.     # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.15,0.3,0.48,0.6,0.75]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jumpv      = np.array([200,100,63,50,40])    # Vetor com Diferentes Valores de Jump
    jump       = jumpv[0]                        # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 2                               # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 1                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste3_ref1:
# Configuração Non-Absorbing para o Teste 3
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    nptx   = 1001     # Número de Pontos Direção X
    npty   = 1001     # Número de Pontos Direção Y
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
    tn     = 2000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
    f0     = 0.02    # Frequência da Fonte (1 Hz = 0.001 kHz) #min de 0.02 até 0.05
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 1000.    # Posição da Fonte em X
    yposf  = 800.     # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  = 1000                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.15])      # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jump       = 1000                            # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 4                               # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 1                               # Boundary Type    
    ftype      = 0                               # Source type                                                     
#==============================================================================

#==============================================================================
class teste4:
# Configuração Non-Absorbing para o Teste 4
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    nptx   = 401     # Número de Pontos Direção X
    npty   = 311     # Número de Pontos Direção Y
    x0     = 25000.  # Ponto Inicial Direção X
    x1     = 45000.  # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     =  9920.  # Ponto Final Direção Y
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
    xposf  = 35150    # Posição da Fonte em X
    yposf  =   32.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  32.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.0875,0.21875,0.35,0.4375,0.525]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jumpv      = np.array([380,150,95,76,63])    # Vetor com Diferentes Valores de Jump
    jump       = jumpv[0]                        # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 2                               # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#==============================================================================

#==============================================================================
class teste4_ref1:
# Configuração Non-Absorbing para o Teste 4
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    nptx   = 2001    # Número de Pontos Direção X
    npty   = 1551    # Número de Pontos Direção Y
    x0     = 25000.  # Ponto Inicial Direção X
    x1     = 45000.  # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     =  9920.  # Ponto Final Direção Y
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
    xposf  = 35150    # Posição da Fonte em X
    yposf  =   32.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  32.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.21875]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jumpv      = np.array([750])    # Vetor com Diferentes Valores de Jump
    jump       = jumpv[0]                        # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                
#==============================================================================