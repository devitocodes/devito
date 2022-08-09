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
import matplotlib.ticker       as mticker    
from   scipy.interpolate       import CubicSpline
from   mpl_toolkits.mplot3d    import Axes3D
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   matplotlib              import cm
from   matplotlib.animation    import FFMpegWriter
from   matplotlib              import ticker
import matplotlib              as mpl
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
# Lendo Resultados Globais e Salvando Dados
#==============================================================================
type_test = 7
vstrategy = [0,1]
nstrategy = len(vstrategy)

if(type_test==1): locsave = 'teste1/'
if(type_test==2): locsave = 'teste2/'
if(type_test==3): locsave = 'teste3/'
if(type_test==4): locsave = 'teste4/'
if(type_test==5): locsave = 'teste5/'
if(type_test==6): locsave = 'teste6/'
if(type_test==7): locsave = 'teste7/'

verrol2glob     = []
infglob         = []
vhxglob         = []
vhzglob         = []
vtimeglob       = []
vpropertiesglob = []

for m0 in range(0,nstrategy):

    strategy = int(vstrategy[m0])
    
    if(strategy==0): locsavestr = 'strategy0/'
    if(strategy==1): locsavestr = 'strategy1/'

    verrol2glob.append(np.load("data_save/%s%sverrol2glob.npy"%(locsave,locsavestr),allow_pickle=True))  
    infglob.append(np.load("data_save/%s%sinfglob.npy"%(locsave,locsavestr),allow_pickle=True))  
    vhxglob.append(np.load("data_save/%s%svhxglob.npy"%(locsave,locsavestr),allow_pickle=True))  
    vhzglob.append(np.load("data_save/%s%svhzglob.npy"%(locsave,locsavestr),allow_pickle=True))  
    vtimeglob.append(np.load("data_save/%s%svtimeglob.npy"%(locsave,locsavestr),allow_pickle=True))  
    vpropertiesglob.append(np.load("data_save/%s%svpropertiesglob.npy"%(locsave,locsavestr),allow_pickle=True)) 

nverrol2glob     = len(verrol2glob[0])
ninfglob         = len(infglob[0])
nvhxglob         = len(vhxglob[0])
nvhzglob         = len(vhzglob[0])
nvtimeglob       = len(vtimeglob[0])
nvpropertiesglob = len(vpropertiesglob[0])
vhx              = vhxglob[0][0,:]
vhz              = vhzglob[0][0,:]
nexpglob         = nverrol2glob
nvhx             = vhx.shape[0]
nvhz             = vhz.shape[0]
#==============================================================================

#==============================================================================
# Plot Map 2
#==============================================================================
def ploterrormap2(Merrors,hx,vpoints,vsou_teste,vstrategy,type_test,locsave):
      
    plot.figure(figsize = (20,10))
    plot.suptitle("Square Rhombus Adapted \n Mapa do Erro na Avaliação do Laplaciano no Teste %d - hx=hz = %.3e"%(type_test,hx))

    grid = plot.GridSpec(1,2,wspace=0.4,hspace=0.2)
    position_plot_listx = np.array([0,0,1])
    position_plot_listy = np.array([0,1,0])
    nsub                = len(position_plot_listx)
    nvstrategy          = len(vstrategy)
    
    vmax = np.amax(Merrors[np.isfinite(Merrors)])
    vmin = np.amin(Merrors[np.isfinite(Merrors)])    
    
    for k in range(0,nvstrategy):
            
        xpos     = int(position_plot_listx[k])
        ypos     = int(position_plot_listy[k])    
        plot.subplot(grid[xpos,ypos])

        U = Merrors[k,:,:]
    
        x0 = vsou_teste[0]
        x1 = vsou_teste[-1]
        z0 = vpoints[0]
        z1 = vpoints[-1]
    
        extent = [z0,z1,x1,x0]    
        fig = plot.imshow(U,cmap=cm.rainbow,extent=extent,interpolation='nearest',aspect='auto',norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        
        strategy = int(vstrategy[k])
        
        if(strategy==0): plot.title('Number of Equations <= Number of Coefficients in the Linear System',fontsize=8,fontdict={'horizontalalignment': 'center'})
        if(strategy==1): plot.title('Number of Equations >= Number of Coefficients in the Linear System',fontsize=8,fontdict={'horizontalalignment': 'center'})
        
        plot.ylabel('[Order]')
        plot.xlabel('[Number of Extra Points]')
        
        plot.grid()
        ax = plot.gca()
        
        ax.xaxis.set_major_locator(plot.MaxNLocator(len(vpoints)))
        ax.yaxis.set_major_locator(plot.MaxNLocator(len(vsou_teste)))

        ticks = np.linspace(vmin, vmax, 20, endpoint=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.025)
        cbar = plot.colorbar(fig, cax=cax, format='%.2e')
        cbar.set_label('[L2 Error]')
        plot.draw()
        plot.show()

    plot.savefig('figures/%smap_erro_group_teste=%d_hx=%.3e.png'%(locsave,type_test,hx),dpi=100)
    plot.show()
    plot.close()
#==============================================================================

#==============================================================================
# # Plot Results Group
#==============================================================================
vsou_teste    = [2,4,6,8,10]
nvsou_teste   = len(vsou_teste)
nlin          = nvsou_teste
ncol          = int(np.amax(infglob[0][:,2]))+1
Merro2        = np.zeros((nstrategy,nlin,ncol))
Merro2[:,:,:] = np.nan
vpoints       = np.arange(0,ncol)

for m3 in range(0,nvhx):
    
    hx = vhx[m3]
    
    for m0 in range(0,nstrategy):
    
        for m1 in range(0,nlin):
    
            sou_teste = vsou_teste[m1]
            sou_teste = int(sou_teste)
            contcol   = 0 

            for m2 in range(0,nexpglob):
    
                posloc         = int(infglob[m0][m2,0])
                souloc         = int(infglob[m0][m2,1])
                extrapointsloc = int(infglob[m0][m2,2])
    
                if(souloc==sou_teste):
        
                    avalue                = verrol2glob[m0][m2,m3]
                    Merro2[m0,m1,contcol] = avalue
                    contcol               = contcol + 1
                    
    P3 = ploterrormap2(Merro2,hx,vpoints,vsou_teste,vstrategy,type_test,locsave)
#==============================================================================