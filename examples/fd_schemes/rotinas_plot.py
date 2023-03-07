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
import matplotlib.ticker       as mticker    
from   scipy.interpolate       import CubicSpline
from   mpl_toolkits.mplot3d    import Axes3D
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   matplotlib              import cm
from   matplotlib.animation    import FFMpegWriter
from   matplotlib              import ticker
from   matplotlib.colors       import LogNorm
import matplotlib.tri          as tri
#==============================================================================

#==============================================================================
# Plot do Deslocamento
#==============================================================================
def graph2d(U,teste):

    x0 = teste.x0
    x1 = teste.x1
    y0 = teste.y0
    y1 = teste.y1

    sou     = teste.sou    
    npesos  = teste.npesos
    mvalue  = int(sou/2)
    nvalue  = teste.nvalue
    wauthor = teste.wauthor
    wtype   = teste.wtype
    exttrap = teste.exttrap
    
    if(npesos==0): wauthorw = 'Taylor Stencil - Cross-Line'
            
    if((npesos==1)and(wauthor==1)and(wtype==1)): wauthorw = 'Yang Liu - 2013 - Cross-Line'

    if((npesos==1)and(wauthor==2)and(wtype==1)): wauthorw = 'Yang Liu - Mrinal Sen - 2009 - Cross-Line'

    if((npesos==1)and(wauthor==3)and(wtype==1)): wauthorw = 'Enjiang Wang - Yang Liu - Mrinal Sen - 2016 - Cross-Rhombus TE'

    if((npesos==1)and(wauthor==3)and(wtype==2)): wauthorw = 'Enjiang Wang - Yang Liu - Mrinal Sen - 2016 - Cross-Rhombus LS'
    
    if((npesos==1)and(wauthor==4)and(wtype==1)): wauthorw = 'Taylor Stencil - Cross-Square'
    
    plot.figure(figsize = (14,10))
    fscale =  10**(-3)

    scale = np.amax(U)/10
    extent = [fscale*x0,fscale*x1, fscale*y1, fscale*y0]
    fig = plot.imshow(np.transpose(U), vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent,interpolation='bicubic', aspect='auto')

    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.axis('equal')
    
    if(wauthor!=4): plot.title('%s - M = %d and N = %d'%(wauthorw,mvalue,nvalue),fontsize=10)
    if(wauthor==4): plot.title('%s - M = %d and Extra Points = %d'%(wauthorw,mvalue,exttrap),fontsize=10)
    
    plot.grid()
    ax = plot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.025)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar = plot.colorbar(fig, cax=cax, format='%.2e')
    cbar.locator = tick_locator
    cbar.update_ticks()
    plot.draw()
    
    if(wauthor!=4): plot.savefig('figures/disp_%d%d%d%d%d.png'%(npesos,wauthor,wtype,mvalue,nvalue),dpi=100)
    if(wauthor==4): plot.savefig('figures/disp_%d%d%d%d%d.png'%(npesos,wauthor,wtype,mvalue,exttrap),dpi=100)
    
    plot.show()
    plot.close()
#==============================================================================

#==============================================================================
# Plot dos Receivers
#==============================================================================
def graph2drec(rec,teste):
    
    x0 = teste.x0
    x1 = teste.x1
    y0 = teste.y0
    y1 = teste.y1
    t0 = teste.t0
    tn = teste.tn
 
    sou     = teste.sou    
    npesos  = teste.npesos
    mvalue  = int(sou/2)
    nvalue  = teste.nvalue
    wauthor = teste.wauthor
    wtype   = teste.wtype
    exttrap = teste.exttrap
    
    if(npesos==0): wauthorw = 'Taylor Stencil - Cross-Line'
            
    if((npesos==1)and(wauthor==1)and(wtype==1)): wauthorw = 'Yang Liu - 2013 - Cross-Line'

    if((npesos==1)and(wauthor==2)and(wtype==1)): wauthorw = 'Yang Liu - Mrinal Sen - 2009 - Cross-Line'

    if((npesos==1)and(wauthor==3)and(wtype==1)): wauthorw = 'Enjiang Wang - Yang Liu - Mrinal Sen - 2016 - Cross-Rhombus TE'

    if((npesos==1)and(wauthor==3)and(wtype==2)): wauthorw = 'Enjiang Wang - Yang Liu - Mrinal Sen - 2016 - Cross-Rhombus LS'
    
    if((npesos==1)and(wauthor==4)and(wtype==1)): wauthorw = 'Taylor Stencil - Cross-Square'
   
    plot.figure(figsize = (14,10))
    fscale =  10**(-3)    
    scale = np.amax(rec)/50.
    extent = [fscale*x0,fscale*x1, fscale*tn, fscale*t0]
    fig = plot.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
        
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
    plot.axis('equal')
    if(wauthor!=4): plot.title('%s - M = %d and N = %d'%(wauthorw,mvalue,nvalue),fontsize=10)
    if(wauthor==4): plot.title('%s - M = %d and Extra Points = %d'%(wauthorw,mvalue,exttrap),fontsize=10)
    plot.grid()
    ax = plot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.025)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar = plot.colorbar(fig, cax=cax, format='%.2e')
    cbar.locator = tick_locator
    cbar.update_ticks()
    if(wauthor!=4): plot.savefig('figures/rec_%d%d%d%d%d.png'%(npesos,wauthor,wtype,mvalue,nvalue),dpi=100)
    if(wauthor==4): plot.savefig('figures/rec_%d%d%d%d%d.png'%(npesos,wauthor,wtype,mvalue,exttrap),dpi=100)
    plot.show()
    plot.close()
#==============================================================================

#==============================================================================
# Plot Velocidades
#==============================================================================
def graph2dvel(vel,teste):
    
    x0 = teste.x0
    x1 = teste.x1
    y0 = teste.y0
    y1 = teste.y1        
  
    plot.figure(figsize = (14,10))
    fscale =  10**(-3)
           
    scale  = np.amax(vel)
    extent = [fscale*x0,fscale*x1, fscale*y1, fscale*y0]
    fig = plot.imshow(np.transpose(vel), vmin=0,vmax=scale, cmap=cm.jet, extent=extent)
          
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.title('Velocity Profile',fontsize=10)
    plot.grid()
    ax = plot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.025)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar = plot.colorbar(fig, cax=cax, format='%.2e')
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.set_label('Velocity [km/s]')
    plot.savefig('figures/vel_map.png')
    plot.show()
    plot.close()
#==============================================================================
