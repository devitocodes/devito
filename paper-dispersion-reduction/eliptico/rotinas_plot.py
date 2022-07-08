#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================

#==============================================================================
# Módulos Importados do Python / Devito / Examples
#==============================================================================

#==============================================================================
# Python Modules and Imports
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
from   numpy                   import linalg              as la  
#==============================================================================

#==============================================================================
# Plot Solução
#==============================================================================
def plotsol(F,DF2exact,DF2num,properties):

    teste      = properties[0]
    teste_type = teste.teste_type
    sou        = teste.sou
    hxv        = teste.hx
    hzv        = teste.hz  
    nptx       = teste.nptx
    nptz       = teste.nptz
    erro2      = properties[1]
    erro2_rel  = properties[2]
    X          = teste.X
    Z          = teste.Z

    Xgrid, Zgrid = np.meshgrid(X,Z)
    fig = plot.figure(figsize=(14,6))
    fig.suptitle("Avaliação Gráfica do Laplaciano no Teste %d - Space Order = %d \n \n \
                 hx = %.2f - hz = %.2f - nptx = %d - nptz = %d \n \n \
                 Erro L2 = %.3e - Erro L2 Relativo = %.3e"
                 %(teste_type,sou,hxv,hzv,nptx,nptz,erro2,erro2_rel), fontsize=10, fontdict={'horizontalalignment': 'center'})
    
    fig.tight_layout()

    field = F
    field_name = 'F'
    ax    = fig.add_subplot(1,3,1,projection='3d')
    fig1 = ax.plot_surface(Xgrid,Zgrid,field,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
    zmin = np.amin(field)    
    zmax = np.amax(field)    
    ax.set_zlim(zmin,zmax)
    fig.colorbar(fig1, shrink=5, aspect=2.5,pad=0.2)
    ax.set_title('%s'%field_name);
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('F(X,Z)')

    field = DF2exact
    field_name = 'DF2exact'
    ax    = fig.add_subplot(1,3,2,projection='3d')
    fig1 = ax.plot_surface(Xgrid,Zgrid,field,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
    zmin = np.amin(field)    
    zmax = np.amax(field)    
    ax.set_zlim(zmin,zmax)
    fig.colorbar(fig1, shrink=5, aspect=2.5,pad=0.2)
    ax.set_title('%s'%field_name);
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('DF2(X,Z)')
    
    field = DF2num
    field_name = 'DF2num'
    ax    = fig.add_subplot(1,3,3,projection='3d')
    fig1 = ax.plot_surface(Xgrid,Zgrid,field,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
    zmin = np.amin(field)    
    zmax = np.amax(field)    
    ax.set_zlim(zmin,zmax)
    fig.colorbar(fig1, shrink=5, aspect=2.5,pad=0.2)
    ax.set_title('%s'%field_name);
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('DF2(X,Z)')
    
    plot.savefig('figures/teste%d.png'%teste_type,dpi=100)
    
    plot.show()
    #plot.close()
#==============================================================================

#==============================================================================
# Plot 1 do Erro da Solução
#==============================================================================
def ploterro1(F,DF2exact,DF2num,properties):
    
    U          = DF2exact - DF2num
    teste      = properties[0]
    teste_type = teste.teste_type
    sou        = teste.sou
    hxv        = teste.hx
    hzv        = teste.hz  
    nptx       = teste.nptx
    nptz       = teste.nptz
    erro2      = properties[1]
    erro2_rel  = properties[2]
    X          = teste.X
    Z          = teste.Z
    x0         = teste.x0
    x1         = teste.x1
    z0         = teste.z0
    z1         = teste.z1
    
    plot.figure(figsize = (14,10))

    scale = np.amax(U)
    
    extent = [x0,x1,z0,z1]
    #fig = plot.imshow(np.transpose(U), vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent,interpolation='bilinear')
    fig = plot.contourf(np.transpose(U), vmin=-scale, vmax=scale, cmap=cm.seismic, levels=64, extent=extent)

    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.axis('equal')

    plot.title("Avaliação Gráfica do Erro do Laplaciano no Teste %d - Space Order = %d \n \n \
              hx = %.2f - hz = %.2f - nptx = %d - nptz = %d \n \n \
              Erro L2 = %.3e - Erro L2 Relativo = %.3e"
              %(teste_type,sou,hxv,hzv,nptx,nptz,erro2,erro2_rel), fontsize=10, fontdict={'horizontalalignment': 'center'})
    
    plot.grid()
    ax = plot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.025)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar = plot.colorbar(fig, cax=cax, format='%.2e')
    cbar.locator = tick_locator
    cbar.update_ticks()
    plot.draw()
    
    plot.savefig('figures/erro_teste%d.png'%teste_type,dpi=100)

    plot.show()
    #plot.close()
#==============================================================================

#==============================================================================
# Plot 2 do Erro da Solução
#==============================================================================
def ploterro2(F,DF2exact,DF2num,properties):
    
    U            = DF2exact - DF2num
    teste        = properties[0]
    teste_type   = teste.teste_type
    sou          = teste.sou
    hxv          = teste.hx
    hzv          = teste.hz  
    nptx         = teste.nptx
    nptz         = teste.nptz
    erro2        = properties[1]
    erro2_rel    = properties[2]
    X            = teste.X
    Z            = teste.Z
    x0           = teste.x0
    x1           = teste.x1
    z0           = teste.z0
    z1           = teste.z1
    Xgrid, Zgrid = np.meshgrid(X,Z)
      
    scale = np.amax(U)
    
    fig = plot.figure(figsize = (14,10))
    field = U
    ax = plot.axes(projection='3d')
    fig1 = ax.plot_surface(Xgrid,Zgrid,field,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
    zmin = np.amin(field)    
    zmax = np.amax(field)    
    ax.set_zlim(zmin,zmax)
    fig.colorbar(fig1, shrink=5, aspect=2.5,pad=0.2)
    ax.set_title("Avaliação Gráfica do Erro do Laplaciano no Teste %d - Space Order = %d \n \n \
            hx = %.2f - hz = %.2f - nptx = %d - nptz = %d \n \n \
            Erro L2 = %.3e - Erro L2 Relativo = %.3e"
            %(teste_type,sou,hxv,hzv,nptx,nptz,erro2,erro2_rel), fontsize=10, fontdict={'horizontalalignment': 'center'})
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Erro(X,Z)')    
    plot.savefig('figures/erro_teste%d.png'%teste_type,dpi=100)

    plot.show()
    #plot.close()
#==============================================================================

#==============================================================================
# Calcula Erros - 1
#==============================================================================
def calc_erro1(DF2exact,DF2num,pair_choo,properties,allow_print):

    teste     = properties[0]
    normtype  = teste.normtype
    X         = teste.X
    Z         = teste.Z
    hxv       = teste.hx
    hzv       = teste.hz
    nptx      = teste.nptx
    nptz      = teste.nptz
    
    erro2     = np.sqrt(hxv*hzv)*la.norm(np.reshape(DF2exact-DF2num,-1),normtype)
    erro2_rel = la.norm(np.reshape(DF2exact-DF2num,-1),normtype)/la.norm(np.reshape(DF2exact,-1),normtype)
    
    xchoo = pair_choo[0]
    zchoo = pair_choo[1]
    
    for i in range(0,nptx):
        
        for j in range(0,nptz):
            
            if(X[i]==xchoo): xpos = i
            if(Z[j]==zchoo): zpos = j

    pointerror = np.abs(DF2exact[xpos,zpos]-DF2num[xpos,zpos])
    
    if(allow_print==1):
    
        print('')
        print('==========================================')
        print('Erro L2: %.3e'%erro2)
        print('')
        print('Erro L2 Relativo: %.3e'%erro2_rel)
        print('==========================================')
        print('')

        print('')
        print('==========================================================')
        print('Erro Pontual: (x = %.3f , z = %.3f) - Erro = %.3e'%(xchoo,zchoo,pointerror))
        print('==========================================================')
        print('')
    
    return erro2, erro2_rel, pointerror 
#==============================================================================

#==============================================================================
# Calcula Erros - 1
#==============================================================================
def calc_erro2(DF2exact,DF2num,pair_choo,vproperties,allow_print,vhx,vhz,vnptx,vnptz):

    teste     = vproperties[0][0]
    normtype  = teste.normtype
    X         = teste.X
    Z         = teste.Z
    hxv       = teste.hx
    hzv       = teste.hz
    nptx      = teste.nptx
    nptz      = teste.nptz    

    ratiox       = int(vhx[0]/hxv)
    ratioz       = int(vhz[0]/hzv)
    nptxloc      = vnptx[0]
    nptzloc      = vnptz[0]
    DF2exact_red = np.zeros((nptxloc,nptzloc))
    DF2num_red   = np.zeros((nptxloc,nptzloc))
    icont        = 0
    jcont        = 0
        
    for l2 in range(0,nptzloc):
            
        icont = 0
            
        for l1 in range(0,nptxloc):
     
            DF2exact_red[l1,l2] = DF2exact[icont,jcont]
            DF2num_red[l1,l2]   = DF2num[icont,jcont]
    
            icont = icont + ratiox
            
        jcont = jcont + ratioz
            
    erro2     = np.sqrt(vhx[0]*vhz[0])*la.norm(np.reshape(DF2exact_red-DF2num_red,-1),normtype)
    erro2_rel = la.norm(np.reshape(DF2exact_red-DF2num_red,-1),normtype)/la.norm(np.reshape(DF2exact_red,-1),normtype)
    
    xchoo = pair_choo[0]
    zchoo = pair_choo[1]
    
    for i in range(0,nptx):
        
        for j in range(0,nptz):
            
            if(X[i]==xchoo): xpos = i
            if(Z[j]==zchoo): zpos = j

    pointerror = np.abs(DF2exact[xpos,zpos]-DF2num[xpos,zpos])
    
    if(allow_print==1):
    
        print('')
        print('==========================================')
        print('Erro L2: %.3e'%erro2)
        print('')
        print('Erro L2 Relativo: %.3e'%erro2_rel)
        print('==========================================')
        print('')

        print('')
        print('==========================================================')
        print('Erro Pontual: (x = %.3f , z = %.3f) - Erro = %.3e'%(xchoo,zchoo,pointerror))
        print('==========================================================')
        print('')
    
    return erro2, erro2_rel, pointerror
#==============================================================================

#==============================================================================
# Plot Convergencia
#==============================================================================
def plot_convcurve1(vhx,verrol2,vproperties):
    
    plot.figure(figsize = (12,12))

    teste      = vproperties[0][0]
    sou        = teste.sou
    teste_type = teste.teste_type
    wauthor    = teste.wauthor
    wtype      = teste.wtype
    
    plot.grid()
    
    plot.title('Estudo de Convergência do Teste %d com Space Order = %d - Wauthor = %d - Wtype = %d'%(teste_type,sou,wauthor,wtype))
    
    plot.plot(vhx,verrol2,color='blue',label='Erro L2')
    ax = plot.gca()

    sou2 = sou - 2
    if(sou==2): sou2 = 1
    theory2 = [t**sou2 for t in vhx]
    correc2 = 1.2*np.amax(verrol2)
    theory2 = [correc2*th/theory2[0] for th in theory2]        
    plot.plot(vhx,theory2,color='red',label='%d-th Inclination'%sou2)

    theory1 = [t**sou for t in vhx]
    correc1 = 1.2*np.amax(verrol2)
    theory1 = [correc1*th/theory1[0] for th in theory1]        
    plot.plot(vhx,theory1,color='gray',label='%d-th Inclination'%sou)
    
    plot.legend(loc="lower center")

    plot.xlabel('log(hx=hz)')
    plot.xticks(vhx)
    plot.ticklabel_format(axis='x', style='sci', scilimits=(1,5))
    ax.set_xscale('log')

    plot.ylabel('log(L2 Error)')
    plot.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
    ax.set_yscale('log')

    min_value = 0.6*min(np.amin(verrol2),np.amin(theory1),np.amin(theory2))
    max_value = 1.4*max(np.amax(verrol2),np.amax(theory1),np.amax(theory2))
    plot.ylim((min_value,max_value))
    
    plot.savefig('figures/convergencia_teste%d_sou=%d_wauthor=%d_wtype=%d.png'%(teste_type,sou,wauthor,wtype),dpi=100)

    plot.show()
    #plot.close()
#==============================================================================

#==============================================================================
# Plot Convergencia
#==============================================================================
def plot_convcurve2(vhx,verrol20,verrol21,vproperties,vwauthor,vwtype):
    
    plot.figure(figsize = (12,12))

    teste      = vproperties[0][0]
    sou        = teste.sou
    teste_type = teste.teste_type
    
    plot.grid()
    plot.title('Comparação do Estudo de Convergência do Teste %d com Space Order = %d'%(teste_type,sou))
    plot.plot(vhx,verrol20,color='blue',label='Erro L2 - Wauthor = %d - Wtype = %d'%(vwauthor[0],vwtype[0]))
    plot.plot(vhx,verrol21,color='black',label='Erro L2 - Wauthor  = %d - Wtype = %d'%(vwauthor[1],vwtype[1]))
    ax = plot.gca()

    sou2 = sou - 2
    if(sou==2): sou2 = 1
    theory2 = [t**sou2 for t in vhx]
    correc2 = 1.2*max(np.amax(verrol20),np.amax(verrol21))
    theory2 = [correc2*th/theory2[0] for th in theory2]        
    plot.plot(vhx,theory2,color='red',label='%d-th Inclination'%sou2)

    theory1 = [t**sou for t in vhx]
    correc1 = 1.2*max(np.amax(verrol20),np.amax(verrol21))
    theory1 = [correc1*th/theory1[0] for th in theory1]        
    plot.plot(vhx,theory1,color='gray',label='%d-th Inclination'%sou)
    
    plot.legend(loc="lower center")

    plot.xlabel('log(hx=hz)')
    plot.xticks(vhx)
    plot.ticklabel_format(axis='x', style='sci', scilimits=(1,5))
    ax.set_xscale('log')

    plot.ylabel('log(L2 Error)')
    plot.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
    ax.set_yscale('log')

    min_value = 0.6*min(np.amin(verrol20),np.amin(verrol21),np.amin(theory1),np.amin(theory2))
    max_value = 1.4*max(np.amax(verrol20),np.amax(verrol21),np.amax(theory1),np.amax(theory2))
    plot.ylim((min_value,max_value))
    
    plot.savefig('figures/comp_convergencia_teste%d_sou=%d.png'%(teste_type,sou),dpi=100)

    plot.show()
    #plot.close()
#==============================================================================