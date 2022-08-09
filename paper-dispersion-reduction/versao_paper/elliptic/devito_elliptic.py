#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================

#==============================================================================
# Módulos Importados do Python / Devito / Examples
#==============================================================================

#==============================================================================
# Python Modules
#==============================================================================
import numpy                   as     np
import sys
import matplotlib.pyplot       as     plot
import time                    as     tm
#==============================================================================

#==============================================================================
# Our Rotines
#==============================================================================
import testes_opt              as ttopt
import rotinas_plot            as rplot
import mfunc                   as mf
import coef_opt                as copt
#==============================================================================

#==============================================================================
# Devito Imports
#==============================================================================
from devito import *
configuration['log-level']='ERROR'
#==============================================================================

#==============================================================================
# Shell Comands
#==============================================================================
import os
os.system('clear')
#==============================================================================

#==============================================================================
plot.close("all")
#==============================================================================

#==============================================================================
# Testes de Leitura de Dados
#==============================================================================
teste = ttopt.teste7
MV    = mf.ellipctdevito(teste)
coef1 = copt.coefopt1(teste)
#==============================================================================

#==============================================================================
# Obtenção de Parâmetros
#==============================================================================
teste_type = teste.teste_type
dtype      = teste.dtype
nptx       = teste.nptx
nptz       = teste.nptz
x0         = teste.x0
x1         = teste.x1 
compx      = teste.compx
z0         = teste.z0
z1         = teste.z1
compz      = teste.compz
hxv        = teste.hx           
hzv        = teste.hz               
X          = teste.X    
Z          = teste.Z
sou        = teste.sou                                
wauthor    = teste.wauthor                                
wtype      = teste.wtype
nvalue     = teste.nvalue
exttrapmax = teste.exttrapmax
exttrap    = teste.exttrap 
nptx_ext   = teste.nptx_ext
nptz_ext   = teste.nptz_ext
Xext       = teste.Xext
Zext       = teste.Zext           
MF         = np.transpose(MV.F)
MF_ext     = np.transpose(MV.F_ext)
MDF2exact  = np.transpose(MV.DF2exact)
normtype   = teste.normtype
properties = [teste]
#==============================================================================

#==============================================================================
# Problem Info
#==============================================================================
print('')
print('==================================================================')
print('Problem =  %d'%(teste_type))
print('sou = %d - nptx = %d - nptz = %d - hx = %.2f - hy = %.2f'%(sou,nptx,nptz,hxv,hzv))
print('==================================================================')
print('')
#==============================================================================

#==============================================================================
# Definição de Vetores Devito
#==============================================================================
origin  = (x0,z0)       
extent  = (compx,compz)
shape   = (nptx,nptz)   
spacing = (hxv,hzv)     

class d0domain(SubDomain):
    name = 'd0'
    def define(self, dimensions):
        x, z = dimensions
        return {x: x, z: z}
d0_domain = d0domain()
    
grid = Grid(origin=origin,extent=extent,shape=shape,subdomains=(d0_domain),dtype=dtype)
#==============================================================================

#==============================================================================
# Variváveis Simbólicas
#==============================================================================
(hx,hz)    = grid.spacing_map  
(x, z)     = grid.dimensions    
#==============================================================================

#==============================================================================
# Construção do Laplaciano
#==============================================================================
F_ext                     = Function(name="F_ext",grid=grid,space_order=sou,staggered=NODE,dtype=dtype)
F_ext.data_with_halo[:,:] = np.transpose(MF_ext[:,:])

Txx,Tyy,mcoef = coef1.calccoef(wauthor,wtype,sou)    
new_laplace, contcoef = coef1.eqconstuct1(mcoef,F_ext,x,z)

print('')
print('==========================================')
expectcoef = (sou+1)**2
print("Coeficientes Não Nulos Esperados  = %d" %expectcoef)
print("Coeficientes Não Nulos Calculados = %d" %contcoef)
print('==========================================')
print('')

DF2num = Function(name="DF2num",grid=grid,space_order=sou,staggered=NODE)
eq0    = Eq(DF2num,new_laplace,subdomain = grid.subdomains['d0'])

DF2exact           = Function(name="DF2exact",grid=grid,space_order=sou,staggered=NODE,dtype=dtype)
DF2exact.data[:,:] = np.transpose(MDF2exact[:,:])

F                  = Function(name="F",grid=grid,space_order=sou,staggered=NODE,dtype=dtype)
F.data[:,:]        = np.transpose(MF[:,:])

op0   = Operator([eq0],subs=grid.spacing_map)
start = tm.time()
op0()
end       = tm.time()
time_exec = end - start

print('')
print('==========================================')
print("Tempo de Execuação = %.3f s" %time_exec)
print('==========================================')
print('')
#==============================================================================

#==============================================================================
# Analise do Erro
#==============================================================================
xchoo       = teste.X[int(nptx/2)]
zchoo       = teste.Z[int(nptz/2)]
pair_choo   = (xchoo,zchoo)
allow_print = 1

erro2, erro2_rel, pointerror = rplot.calc_erro1(DF2exact.data,DF2num.data,pair_choo,properties,allow_print)
properties.append(erro2)
properties.append(erro2_rel)

print('')
print('=================================================')
print("Erro L2                           = %.3e" %erro2)
print("Coeficientes Não Nulos Calculados = %d" %contcoef)
print('=================================================')
print('')
#==============================================================================

#==============================================================================
# Plots de Interesse
#==============================================================================
#P1 = rplot.plotsol(np.transpose(F.data),np.transpose(DF2exact.data),np.transpose(DF2num.data),properties)
#P2 = rplot.ploterro1(F,DF2exact.data,DF2num.data,properties)
#P3 = rplot.ploterro2(F.data,DF2exact.data,DF2num.data,properties)
#==============================================================================