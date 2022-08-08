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
# Configuracoes para Diferentes nptx e nptz
#==============================================================================
cont_test        = 0
wauthor          = 3
wtype            = 1
verrol2glob      = []
infglob          = [] 
vhxglob          = []
vhzglob          = []
vtimeglob        = []
vpropertiesglob  = []
#vorders          = [2,4,6,8,10,12,14,16,18,20]
vorders          = [2,4,6,8,10]  
nvorders         = len(vorders)
vnptx            = [101,201,401,801,1601,3201]
vnptz            = [101,201,401,801,1601,3201]
ndxnz            = len(vnptx)
#==============================================================================

#==============================================================================
# Looping over order
#==============================================================================
for m0 in range(0,nvorders): 
#==============================================================================    

#==============================================================================    
    sou         = vorders[m0]   
    mvalue      = int(sou/2)
    exttrapmax  = int(0.5*(mvalue**2+mvalue))
    vexttrap    = np.arange(0,exttrapmax+1)    
    nvexttrap   = vexttrap.shape[0]
#==============================================================================    

#==============================================================================
# Looping over order
#==============================================================================
    for m1 in range(0,nvexttrap): 
#==============================================================================    

#==============================================================================    
        exttrap     = int(vexttrap[m1])
        verrol2     = []
        verrol2_rel = []
        vproperties = []
        vhx         = []
        vhz         = []
        vtime       = []
        info        = [cont_test,sou,exttrap]
        cont_test   = cont_test + 1
#==============================================================================    

#==============================================================================
# Looping over nptx and nptz
#==============================================================================                
        for m2 in range(0,ndxnz): 
#==============================================================================

#==============================================================================
# Carregando Teste
#==============================================================================
            teste    = ttopt.teste7
            strategy = 1
#==============================================================================

#==============================================================================
# Obtenção de Parâmetros
#==============================================================================
            teste_type             = teste.teste_type
            dtype                  = teste.dtype
            teste.nptx             = vnptx[m2] 
            nptx                   = teste.nptx
            teste.nptz             = vnptz[m2]
            nptz                   = teste.nptz
            x0                     = teste.x0
            x1                     = teste.x1 
            compx                  = teste.compx
            z0                     = teste.z0
            z1                     = teste.z1
            compz                  = teste.compz
            teste.hx               = (x1-x0)/(nptx-1)           
            hxv                    = teste.hx
            teste.hz               = (z1-z0)/(nptz-1)               
            hzv                    = teste.hz
            teste.X                = np.linspace(x0,x1,nptx)    
            X                      = teste.X
            teste.Z                = np.linspace(z0,z1,nptz)
            Z                      = teste.Z
            teste.sou              = sou
            sou                    = teste.sou
            teste.wauthor          = wauthor
            wauthor                = teste.wauthor
            teste.wtype            = wtype
            wtype                  = teste.wtype
            nvalue                 = teste.nvalue
            teste.exttrapmax       = exttrapmax
            exttrapmax             = teste.exttrapmax
            teste.exttrap          = exttrap 
            exttrap                = teste.exttrap 
            teste.nptx_ext         = nptx + 2*sou
            nptx_ext               = teste.nptx_ext    
            teste.nptz_ext         = nptz + 2*sou
            nptz_ext               = teste.nptz_ext
            teste.Xext             = np.linspace(x0-sou*hxv,x1+sou*hxv,nptx_ext)
            teste.Zext             = np.linspace(z0-sou*hzv,z1+sou*hzv,nptz_ext)
            Xext                   = teste.Xext
            Zext                   = teste.Zext           
            normtype               = teste.normtype
            properties             = [teste]
#==============================================================================

#==============================================================================
# Carregando Matrizes e Stencil
#==============================================================================
            MV         = mf.ellipctdevito(teste)
            coef1      = copt.coefopt1(teste)
            MV         = mf.ellipctdevito(teste)    
            MF         = MV.F
            MF_ext     = MV.F_ext
            MDF2exact  = MV.DF2exact
#==============================================================================

#==============================================================================
# Problem Info
#==============================================================================
            print('')
            print('==================================================================')
            print('Problem =  %d'%(teste_type))
            print('sou = %d - nptx = %d - nptz = %d - hx = %.3e - hy = %.3e'%(sou,nptx,nptz,hxv,hzv))
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
            F_ext.data_with_halo[:,:] = MF_ext[:,:]

            Txx,Tyy,mcoef         = coef1.calccoef(wauthor,wtype,sou)    
            new_laplace, contcoef = coef1.eqconstuct1(mcoef,F_ext,x,z)

            DF2num = Function(name="DF2num",grid=grid,space_order=sou,staggered=NODE)
            eq0    = Eq(DF2num,new_laplace,subdomain = grid.subdomains['d0'])

            DF2exact           = Function(name="DF2exact",grid=grid,space_order=sou,staggered=NODE,dtype=dtype)
            DF2exact.data[:,:] = MDF2exact[:,:]

            F                  = Function(name="F",grid=grid,space_order=sou,staggered=NODE,dtype=dtype)
            F.data[:,:]        = MF[:,:]

            op0   = Operator([eq0],subs=grid.spacing_map)
            start = tm.time()
            op0()
            end       = tm.time()
            time_exec = end - start
            info.append(time_exec)
            
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

            if(m2>0):
        
                erro2, erro2_rel, pointerror = rplot.calc_erro2(DF2exact.data,DF2num.data,pair_choo,vproperties,allow_print,vhx,vhz,vnptx,vnptz)
        
            else:
        
                erro2, erro2_rel, pointerror = rplot.calc_erro1(DF2exact.data,DF2num.data,pair_choo,properties,allow_print)

            properties.append(erro2)
            properties.append(erro2_rel)
#==============================================================================

#==============================================================================
# Salvando Resultados Locais
#==============================================================================
            verrol2.append(erro2)
            verrol2_rel.append(erro2_rel)
            vhx.append(hxv)
            vhz.append(hzv)
            vtime.append(time_exec)
            vproperties.append(properties)
#==============================================================================

#==============================================================================
# Salvando Resultados Globais
#==============================================================================
        verrol2glob.append(verrol2)
        infglob.append(info)
        vhxglob.append(vhx)
        vhzglob.append(vhz)
        vtimeglob.append(vtime)
        vpropertiesglob.append(vproperties)
#==============================================================================
        
#==============================================================================
# Salvando Resultados Globais
#==============================================================================
if(teste_type==1): locsave = 'teste1/'
if(teste_type==2): locsave = 'teste2/'
if(teste_type==3): locsave = 'teste3/'
if(teste_type==4): locsave = 'teste4/'
if(teste_type==5): locsave = 'teste5/'
if(teste_type==6): locsave = 'teste6/'
if(teste_type==7): locsave = 'teste7/'

if(strategy==0): locsavestr = 'strategy0/'
if(strategy==1): locsavestr = 'strategy1/'
if(strategy==2): locsavestr = 'strategy2/'

np.save("data_save/%s%sverrol2glob"%(locsave,locsavestr),verrol2glob)  
np.save("data_save/%s%sinfglob"%(locsave,locsavestr),infglob)  
np.save("data_save/%s%svhxglob"%(locsave,locsavestr),vhxglob)  
np.save("data_save/%s%svhzglob"%(locsave,locsavestr),vhzglob)  
np.save("data_save/%s%svtimeglob"%(locsave,locsavestr),vtimeglob)  
np.save("data_save/%s%svpropertiesglob"%(locsave,locsavestr),vpropertiesglob)  
#==============================================================================