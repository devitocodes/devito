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
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    velmode       = 1                                                      # Seleciona Módulo de Velocidade
    nptx          = 101                                                    # Número de Pontos Direção X
    npty          = 101                                                    # Número de Pontos Direção Y
    x0            =    0.                                                  # Ponto Inicial Direção X
    x1            = 3000.                                                  # Ponto Final Direção X 
    compx         = x1-x0                                                  # Comprimento do Domínio em X
    y0            =    0.                                                  # Ponto Inicial Direção Y
    y1            = 3000.                                                  # Ponto Final Direção Y
    compy         = y1-y0                                                  # Comprimento do Domínio em Y
    hx            = (x1-x0)/(nptx-1)                                       # Delta x
    hy            = (y1-y0)/(npty-1)                                       # Delta y    
    X0            = np.linspace(x0,x1,nptx)                                # Malha Direção X
    Y0            = np.linspace(y0,y1,npty)                                # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)                                     # Grid Auxiliar X0 x Y0 
    t0            = 0.                                                     # Tempo Inicial da Simulação em Milisegundos
    tn            = 900.                                                   # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)
    f0            = 0.03                                                   # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte        = 1                                                      # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf         = 1500.                                                  # Posição da Fonte em X
    yposf         = 1500.                                                  # Posição da Fonte em Y
    nrec          = nptx                                                   # Número de Receivers
    nxpos         = np.linspace(x0,x1,nrec)                                # Posição dos Receivers em X
    nypos         = 1500                                                   # Posição dos Receivers em Y
    datainter     = 1                                                      # Interpolação de Dados de Velocidade
    dataintera    = 0                                                      # Interpolação de Dados de Velocidade Artificial
    CFL           = 0.05                                                   # Condição CFL
    tou           = 2                                                      # Time Order Displacement 
    sou           = 2                                                      # Space Order Displacement
    nvalue        = 1                                                      # Second Parameter for Stencils
    exttrapmax    = int(0.5*(int(sou/2)**2+int(sou/2)))                    # Maximum Number of Extra Points 
    exttrap       = np.random.randint(low=0,high=exttrapmax,size=1)[0]     # Random Number of Extra Points
    npesos        = 1                                                      # Allow Different Weights      
    wauthor       = 4                                                      # Weight's Author
    wtype         = 1                                                      # Weight Type
    btype         = 3                                                      # Boundary Type    
    ftype         = 0                                                      # Source Type
#==============================================================================

#==============================================================================
class teste2:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    velmode       = 2                                                      # Seleciona Módulo de Velocidade
    nptx          = 101                                                    # Número de Pontos Direção X
    npty          = 101                                                    # Número de Pontos Direção Y
    x0            =    0.                                                  # Ponto Inicial Direção X
    x1            = 2000.                                                  # Ponto Final Direção X 
    compx         = x1-x0                                                  # Comprimento do Domínio em X
    y0            =    0.                                                  # Ponto Inicial Direção Y
    y1            = 2000.                                                  # Ponto Final Direção Y
    compy         = y1-y0                                                  # Comprimento do Domínio em Y
    hx            = (x1-x0)/(nptx-1)                                       # Delta x
    hy            = (y1-y0)/(npty-1)                                       # Delta y    
    X0            = np.linspace(x0,x1,nptx)                                # Malha Direção X
    Y0            = np.linspace(y0,y1,npty)                                # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)                                     # Grid Auxiliar X0 x Y0 
    t0            = 0.                                                     # Tempo Inicial da Simulação em Milisegundos
    tn            = 540.                                                   # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)
    f0            = 0.025                                                  # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte        = 1                                                      # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf         = 1000.                                                  # Posição da Fonte em X
    yposf         = 800.                                                   # Posição da Fonte em Y
    nrec          = nptx                                                   # Número de Receivers
    nxpos         = np.linspace(x0,x1,nrec)                                # Posição dos Receivers em X
    nypos         = 1000.                                                  # Posição dos Receivers em Y
    datainter     = 1                                                      # Interpolação de Dados de Velocidade
    dataintera    = 0                                                      # Interpolação de Dados de Velocidade Artificial
    CFL           = 0.075                                                  # Condição CFL    
    tou           = 2                                                      # Time Order Displacement 
    sou           = 2                                                      # Space Order Displacement
    nvalue        = 1                                                      # Second Parameter for Stencils
    exttrapmax    = int(0.5*(int(sou/2)**2+int(sou/2)))                    # Maximum Number of Extra Points 
    exttrap       = np.random.randint(low=0,high=exttrapmax,size=1)[0]     # Random Number of Extra Points
    npesos        = 1                                                      # Allow Different Weights      
    wauthor       = 4                                                      # Weight's Author
    wtype         = 1                                                      # Weight Type
    btype         = 1                                                      # Boundary Type    
    ftype         = 0                                                      # Source Type 
#==============================================================================

#==============================================================================
class teste3:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    velmode       = 3                                                      # Seleciona Módulo de Velocidade
    nptx          = 301                                                    # Número de Pontos Direção X
    npty          = 101                                                    # Número de Pontos Direção Y
    x0            =    0.                                                  # Ponto Inicial Direção X
    x1            = 12000.                                                 # Ponto Final Direção X 
    compx         = x1-x0                                                  # Comprimento do Domínio em X
    y0            =    0.                                                  # Ponto Inicial Direção Y
    y1            = 4000.                                                  # Ponto Final Direção Y
    compy         = y1-y0                                                  # Comprimento do Domínio em Y
    hx            = (x1-x0)/(nptx-1)                                       # Delta x
    hy            = (y1-y0)/(npty-1)                                       # Delta y    
    X0            = np.linspace(x0,x1,nptx)                                # Malha Direção X
    Y0            = np.linspace(y0,y1,npty)                                # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)                                     # Grid Auxiliar X0 x Y0 
    t0            = 0.                                                     # Tempo Inicial da Simulação em Milisegundos
    tn            = 1800.                                                  # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)      
    f0            = 0.034                                                  # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte        = 1                                                      # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf         = 6000.                                                  # Posição da Fonte em X
    yposf         = 100.                                                   # Posição da Fonte em Y
    nrec          = nptx                                                   # Número de Receivers
    nxpos         = np.linspace(x0,x1,nrec)                                # Posição dos Receivers em X
    nypos         = 20.                                                    # Posição dos Receivers em Y
    datainter     = 1                                                      # Interpolação de Dados de Velocidade
    dataintera    = 0                                                      # Interpolação de Dados de Velocidade Artificial
    CFL           = 0.0625                                                 # Condição CFL
    tou           = 2                                                      # Time Order Displacement 
    sou           = 2                                                      # Space Order Displacement
    nvalue        = 1                                                      # Second Parameter for Stencils
    exttrapmax    = int(0.5*(int(sou/2)**2+int(sou/2)))                    # Maximum Number of Extra Points 
    exttrap       = np.random.randint(low=0,high=exttrapmax,size=1)[0]     # Random Number of Extra Points
    npesos        = 1                                                      # Allow Different Weights      
    wauthor       = 1                                                      # Weight's Author
    wtype         = 1                                                      # Weight Type
    btype         = 2                                                      # Boundary Type
    ftype         = 0                                                      # Source Type
#==============================================================================

#==============================================================================
class teste4:
#==============================================================================

#==============================================================================
#Parâmetros de Malha e Tempo
#==============================================================================
    velmode       = 4                                                      # Seleciona Módulo de Velocidade
    nptx          = 226                                                    # Número de Pontos Direção X
    npty          = 81                                                     # Número de Pontos Direção Y
    x0            = 4000.                                                  # Ponto Inicial Direção X
    x1            = 13000.                                                 # Ponto Final Direção X 
    compx         = x1-x0                                                  # Comprimento do Domínio em X
    y0            =     0.                                                 # Ponto Inicial Direção Y
    y1            =  3200.                                                 # Ponto Final Direção Y
    compy         = y1-y0                                                  # Comprimento do Domínio em Y
    hx            = (x1-x0)/(nptx-1)                                       # Delta x
    hy            = (y1-y0)/(npty-1)                                       # Delta y    
    X0            = np.linspace(x0,x1,nptx)                                # Malha Direção X
    Y0            = np.linspace(y0,y1,npty)                                # Malha Direção Y
    X0grid,Y0grid = np.meshgrid(X0,Y0)                                     # Grid Auxiliar X0 x Y0 
    t0            = 0.                                                     # Tempo Inicial da Simulação em Milisegundos
    tn            = 1800.                                                  # Tempo Final   da Simulação em Milisegundos (1 Segundo = 1000 Ms)
    f0            = 0.03                                                   # Frequência da Fonte (1 Hz = 0.001 kHz)
    nfonte        = 1                                                      # Número de Fontes  (Para nfonte>1 -> Vetor em xposf e yposf)
    xposf         = 8500                                                   # Posição da Fonte em X
    yposf         = 50.                                                    # Posição da Fonte em Y
    nrec          = nptx                                                   # Número de Receivers
    nxpos         = np.linspace(x0,x1,nrec)                                # Posição dos Receivers em X
    nypos         =  50.                                                   # Posição dos Receivers em Y
    datainter     = 1                                                      # Interpolação de Dados de Velocidade
    dataintera    = 0                                                      # Interpolação de Dados de Velocidade Artificial
    CFL           = 0.0625                                                 # Condição CFL    
    tou           = 2                                                      # Time Order Displacement 
    sou           = 2                                                      # Space Order Displacement
    nvalue        = 1                                                      # Second Parameter for Stencils
    exttrapmax    = int(0.5*(int(sou/2)**2+int(sou/2)))                    # Maximum Number of Extra Points 
    exttrap       = np.random.randint(low=0,high=exttrapmax,size=1)[0]     # Random Number of Extra Points
    npesos        = 1                                                      # Allow Different Weights      
    wauthor       = 1                                                      # Weight's Author
    wtype         = 1                                                      # Weight Type
    btype         = 2                                                      # Boundary Type
    ftype         = 0                                                      # Source Type
#==============================================================================
