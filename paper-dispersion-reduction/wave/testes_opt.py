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
    tn     = 960.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) #min de 0.2s até 2.0s      
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
    sou        = 8                                # Space Order Displacement
    nvalue     = 1                                # Second Parameter for Stencils
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
    npesos     = 1                                # Allow Different Weights      
    wauthor    = 4                                # Weight's Author
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
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 3                               # Boundary Type    
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste2:
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
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
    npesos     = 0                                # Allow Different Weights      
    wauthor    = 1                                # Weight's Author
    wtype      = 1                                # Weight Type
    btype      = 1                                # Boundary Type    
    ftype      = 0                                # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#==============================================================================

#==============================================================================
class teste2_ref1:
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
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
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
    dttype = 1       # Choose dt Value
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
    tn     = 1980.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)      
    f0     = 0.034    # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 6000.    # Posição da Fonte em X
    yposf  =  100.    # Posição da Fonte em Y
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
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
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
    f0     = 0.034    # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 6000.    # Posição da Fonte em X
    yposf  =  100.    # Posição da Fonte em Y
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
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#==============================================================================

#==============================================================================
class teste4:
# Configuração Non-Absorbing para o Teste 4
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 626     # Número de Pontos Direção X - Original 401
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
    f0     = 0.03     # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 35150    # Posição da Fonte em X
    yposf  =   32.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  32.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.125,0.25,0.375,0.5,0.625,0.75]) # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                                # Condição CFL
    jumpv      = np.array([300,150,100,75,60,50])            # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 2                               # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
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
    dttype = 0       # Choose dt Value
    nptx   = 3126    # Número de Pontos Direção X - Original 2001
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
    f0     = 0.03     # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 35150    # Posição da Fonte em X
    yposf  =   32.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  32.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.125])               # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jumpv      = np.array([1500])                # Vetor com Diferentes Valores de Jump
    jump       = jumpv[0]                        # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                
#==============================================================================

#==============================================================================
class teste5:
# Configuração Non-Absorbing para o Teste 4
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 901     # Número de Pontos Direção X - Original 401
    npty   = 321     # Número de Pontos Direção Y
    x0     = 4000.   # Ponto Inicial Direção X
    x1     = 13000.  # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     =  3200.  # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms) # tF = 3000
    f0     = 0.03     # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 8500     # Posição da Fonte em X
    yposf  =   50.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  50.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.25,0.5,0.75,1.0,1.25,1.5])      # Vetor com Diferentes Condições CFL
    CFL        = CFLv[dttype]                                # Condição CFL
    jumpv      = np.array([300,150,100,75,60,50])            # Vetor com Diferentes Valores de Jump
    jump       = jumpv[dttype]                   # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 2                               # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#==============================================================================

#==============================================================================
class teste5_ref1:
# Configuração Non-Absorbing para o Teste 5
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    dttype = 0       # Choose dt Value
    nptx   = 4501    # Número de Pontos Direção X - Original 2001
    npty   = 1601    # Número de Pontos Direção Y
    x0     = 4000.  # Ponto Inicial Direção X
    x1     = 13000.  # Ponto Final Direção X 
    compx  = x1-x0   # Comprimento do Domínio em X
    y0     =    0.   # Ponto Inicial Direção Y
    y1     =  3200.  # Ponto Final Direção Y
    compy  = y1-y0   # Comprimento do Domínio em Y
    hx     = (x1-x0)/(nptx-1)           # Delta x
    hy     = (y1-y0)/(npty-1)           # Delta y    
    X0     = np.linspace(x0,x1,nptx)    # Malha Direção X
    Y0     = np.linspace(y0,y1,npty)    # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)  # Grid Auxiliar X0 x Y0 
    t0     = 0.       # Tempo Inicial da Simulação em Milisegundos
    tn     = 3000.    # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)      
    f0     = 0.03     # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte = 1        # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf  = 8500.    # Posição da Fonte em X
    yposf  =   50.    # Posição da Fonte em Y
    nrec   = nptx     # Número de Receivers
    nxpos  = np.linspace(x0,x1,nrec)   # Posição dos Receivers em X
    nypos  =  50.                      # Posição dos Receivers em Y
    datainter  = 1                     # Interpolação de Dados de Velocidade
    dataintera = 0                     # Interpolação de Dados de Velocidade Artificial
    CFLv       = np.array([0.25])                # Vetor com Diferentes Condições CFL
    CFL        = CFLv[0]                         # Condição CFL
    jumpv      = np.array([1500])                # Vetor com Diferentes Valores de Jump
    jump       = jumpv[0]                        # Intervalo de Dados a Serem Salvos
    tou        = 2                               # Time Order Displacement 
    sou        = 40                              # Space Order Displacement
    nvalue     = 1                               # Second Parameter for Stencils
    exttrapmax = int(0.5*(int(sou/2)**2+int(sou/2))) # Maximum Number of Extra Points 
    exttrap    = np.random.randint(low=0,high=exttrapmax,size=1)[0] # Random Number of Extra Points
    npesos     = 0                               # Allow Different Weights      
    wauthor    = 1                               # Weight's Author
    wtype      = 1                               # Weight Type
    btype      = 2                               # Boundary Type
    ftype      = 0                               # Source type                                                                                                                                                                                                                                                                                                                                                                                                                                                
#==============================================================================