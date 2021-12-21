#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================

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
import testes_opt              as ttopt
import rotinas_plot            as rplot
import macustica               as mc
import coef_opt                as copt
#==============================================================================

#==============================================================================
# Devito Imports
#==============================================================================
from devito import *
#==============================================================================

#==============================================================================
# Devito Examples Imports
#==============================================================================
from   examples.seismic        import TimeAxis
from   examples.seismic        import RickerSource
from   examples.seismic        import Receiver
configuration['log-level']='ERROR'
#==============================================================================

#==============================================================================
plot.close("all")
#==============================================================================

#==============================================================================
# Testes de Leitura de Dados
#==============================================================================
ptype = 1
ref   = 1

if(ref!=0):

    if(ptype==1): teste = ttopt.teste1_ref1
    if(ptype==2): teste = ttopt.teste2_ref1
    if(ptype==3): teste = ttopt.teste3_ref1
    if(ptype==4): teste = ttopt.teste4_ref1

else:

    if(ptype==1): teste = ttopt.teste1
    if(ptype==2): teste = ttopt.teste2
    if(ptype==3): teste = ttopt.teste3
    if(ptype==4): teste = ttopt.teste4
    
MV    = mc.acusdevito(teste)
coef1 = copt.coefopt1(teste,MV)
#==============================================================================

#==============================================================================
# Obtenção de Parâmetros
#==============================================================================
nptx    = teste.nptx    # Número de Pontos Direção X
npty    = teste.npty    # Número de Pontos Direção Y
x0      = teste.x0      # Ponto Inicial da Malha X
y0      = teste.y0      # Ponto Inicial da Malha Y
compx   = teste.compx   # Comprimento Domínio em X
compy   = teste.compy   # Comprimento Domínio em Y 
hxv     = teste.hx      # Delta x
hyv     = teste.hy      # Delta y
t0      = teste.t0      # Tempo Inicial da Simulação em Milisegundos
tn      = teste.tn      # Tempo Final da Simulação em Milisegundos
f0      = teste.f0      # Frequência da Fonte em Khz
nfonte  = teste.nfonte  # Número de Fontes
xposf   = teste.xposf   # Posição da Fonte em X
yposf   = teste.yposf   # Posição da Fonte em Y
nrec    = teste.nrec    # Número de Receivers
nxpos   = teste.nxpos   # Posição dos Receivers em X
nypos   = teste.nypos   # Posição dos Receivers em Y
CFL     = teste.CFL     # Constante de Estabilidade
v       = MV.C0a        # Matriz de Velocidade
jump    = teste.jump    # Intervalo de Plotagem
tou     = teste.tou     # Time Order Displacement 
sou     = teste.sou     # Space Order Displacement 
nvalue  = teste.nvalue  # Second Parameter for Stencils
npesos  = teste.npesos  # Allow Different Weights
wauthor = teste.wauthor # Weight's Author
wtype   = teste.wtype   # Weight Type        
btype   = teste.btype   # Boundary Type
ftype   = teste.ftype   # Source type                 
#==============================================================================

#==============================================================================
# Definição de Vetores Devito
#==============================================================================
origin  = (x0,y0)       
extent  = (compx,compy)
shape   = (nptx,npty)   
spacing = (hxv,hyv)     

class d0domain(SubDomain):
    name = 'd0'
    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: y}
d0_domain = d0domain()
    
grid = Grid(origin=origin,extent=extent,shape=shape,subdomains=(d0_domain),dtype=np.float64)
#==============================================================================

#==============================================================================
# Construção da Malha Temporal
#==============================================================================
vmax  = np.around(np.amax(v),1) 
dtmax = (min(hxv,hyv)*CFL)/(vmax)
ntmax = int((tn-t0)/dtmax)
dt0   = (tn-t0)/(ntmax)
time_range = TimeAxis(start=t0,stop=tn,num=ntmax+1)
nt         = time_range.num - 1
nplot      = mt.ceil(nt/jump) + 1
#==============================================================================
#print(dt0,nt,jump,nplot,hxv,hyv)
#sys.exit()
#==============================================================================
# Variváveis Simbólicas
#==============================================================================
(hx,hy)    = grid.spacing_map  
(x, y)     = grid.dimensions    
time       = grid.time_dim     
t          = grid.stepping_dim 
dt         = grid.stepping_dim.spacing
#==============================================================================

#==============================================================================
# Construção e Posicionamento da Fonte
#==============================================================================
src = RickerSource(name='src',grid=grid,f0=f0,npoint=nfonte,time_range=time_range,staggered=NODE,dtype=np.float64)
src.coordinates.data[:, 0] = xposf
src.coordinates.data[:, 1] = yposf
#==============================================================================

#==============================================================================
# Construção e Posicionamento dos Receivers
#==============================================================================
rec = Receiver(name='rec',grid=grid,npoint=nrec,time_range=time_range,staggered=NODE,dtype=np.float64)
rec.coordinates.data[:, 0] = nxpos
rec.coordinates.data[:, 1] = nypos
#==============================================================================

#==============================================================================
# Construção e Posicionamento dos Receivers Seleionados
#==============================================================================
if(ptype==1):

    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])

if(ptype==2):

    xpositionv  = np.array([4000.0,4000.0,4000.0,6000.0,6000.0,6000.0,8000.0,8000.0,8000.0,])   
    ypositionv  = np.array([2000.0,2500.0,1500.0,3000.0,2000.0,2500.0,1500.0,3000.0,2000.0,2500.0,1500.0,3000.0])    

if(ptype==3):
    
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])

if(ptype==4):
    
    xpositionv  = np.array([30000.0,30000.0,30000.0,40000.0,40000.0,40000.0])
    ypositionv  = np.array([2500.0,5000.0,7500.0,2500.0,5000.0,7500.0,2500.0,5000.0,7500.0])

nrec_select = len(xpositionv)
rec_select  = Receiver(name='rec_select',grid=grid,npoint=nrec_select,time_range=time_range,staggered=NODE,dtype=np.float64)
rec_select.coordinates.data[:, 0] = xpositionv
rec_select.coordinates.data[:, 1] = ypositionv
#==============================================================================

#==============================================================================
# Construção da Equação da Onda com Termo de Fonte
#==============================================================================
u = TimeFunction(name="u",grid=grid,time_order=tou,space_order=sou,staggered=NODE,dtype=np.float64)

vel = Function(name="vel",grid=grid,space_order=2,staggered=NODE,dtype=np.float64)
vel.data[:,:] = v[:,:]

fact = 1/(hxv*hyv)
src_term = src.inject(field=u.forward,expr=fact*1*src*dt**2*vel**2)
rec_term = rec.interpolate(expr=u)
rec_select_term = rec_select.interpolate(expr=u)

if(npesos==0):

    pde0     = Eq(u.dt2 - u.laplace*vel*vel)
    stencil0 = Eq(u.forward, solve(pde0,u.forward),subdomain = grid.subdomains['d0'])

if(npesos==1):
    
    Txx,Tyy,mcoef         = coef1.calccoef(wauthor,wtype,sou,nvalue)    
    new_laplace, contcoef = coef1.eqconstuct(mcoef,u,t,x,y)

    if(wauthor==4 or wauthor==5):
  
        pde0     = new_laplace - u[t-1,x,y]      
        stencil0 = Eq(u[t+1,x,y],pde0,subdomain=grid.subdomains['d0'])
      
    else:
        
        pde0        = Eq(u.dt2 - new_laplace*vel*vel)
        stencil0    = Eq(u.forward, solve(pde0,u.forward),subdomain = grid.subdomains['d0'])
#==============================================================================

#==============================================================================
# Criando Estrutura para Plots Selecionados
#==============================================================================
time_subsampled = ConditionalDimension('t_sub',parent=time,factor=jump)
usave = TimeFunction(name='usave',grid=grid,time_order=tou,space_order=sou,save=nplot,time_dim=time_subsampled,staggered=NODE,dtype=np.float64)
Ug    = np.zeros((nplot,nptx,npty))
#==============================================================================

#==============================================================================
# Construção do Operador de Solução
#==============================================================================
if(btype==1):

    bc = [Eq(u[t+1,0,y],0.),Eq(u[t+1,nptx-1,y],0.),Eq(u[t+1,x,0],0.),Eq(u[t+1,x,npty-1],0.)]
    op = Operator([stencil0] + src_term + bc + rec_term + rec_select_term + [Eq(usave,u.forward)],subs=grid.spacing_map)

if(btype==2):

    bc = [Eq(u[t+1,0,y],0.),Eq(u[t+1,nptx-1,y],0.),Eq(u[t+1,x,npty-1],0.)]
    bc1 = [Eq(u[t+1,x,-k],u[t+1,x,k]) for k in range(1,int(sou/2)+1)]
    op    = Operator([stencil0] + src_term + bc + bc1 + rec_term + rec_select_term + [Eq(usave,u.forward)],subs=grid.spacing_map)

usave.data[:] = 0.
u.data[:]     = 0.

start = tm.time()
op(time=nt,dt=dt0)
end   = tm.time()
time_exec = end - start 

Ug[:] = usave.data[:]
Ug[nplot-1,:,:] = u.data[0,:,:]
#==============================================================================

#==============================================================================
# Plots de Interesse
#==============================================================================
#G1 = rplot.graph2d(u.data[0,:,:],teste,ref)
#R1 = rplot.graph2drec(rec.data,teste,ref)
#V1 = rplot.graph2dvel(v,teste)
S1 = rplot.datasave(teste,rec.data,Ug,rec_select.data,ref)
#==============================================================================

#==============================================================================
print("Tempo de Execuação da Referencia = %.3f s" %time_exec)
#==============================================================================