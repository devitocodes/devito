#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as     np
import sympy                   as     sym
import matplotlib.pyplot       as     plot
import sys
from   sympy.polys.polyfuncs   import interpolate
#==============================================================================

#==============================================================================
# Shell Comands
#==============================================================================
import os
os.system('clear')
#==============================================================================

#==============================================================================
plot.close("all")
plot.rcParams['text.usetex'] = True
#==============================================================================

#==============================================================================
# Yang Liu 2012 Coefficients
#==============================================================================
sys.path.insert(0, './basic_weights')
import basic_weights            as     bw
#==============================================================================

#==============================================================================
# Yang Liu 2012 Coefficients
#==============================================================================
sys.path.insert(0, './coef_yang_liu')
import coef_otm_versao14        as     cotm14
#==============================================================================

#==============================================================================
# Liu and Sen 2009 Coefficients
#==============================================================================
sys.path.insert(0, './coef_liu_sen')
import coef_otm_versao32        as     cotm32
#==============================================================================

#==============================================================================
# Wang, Liu and Sen - Cross/Rombus Stencils - 2016
#==============================================================================
sys.path.insert(0, './coef_wang_liu_sen')
import coef_otm_versao71        as     cotm71
import coef_otm_versao72        as     cotm72
#==============================================================================

#==============================================================================
# Pratical Inputs
#==============================================================================
ptype     = 9
nordem    = 4
mvalue    = int(nordem/2)
nround    = 4
textprint = 0

if(ptype==1):
    
    dx         = 20
    dz         = 20
    vmin       = 1.5
    vmax       = 4.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    
if(ptype==2):
    
    dx         = 10
    dz         = 10
    vmin       = 1.5
    vmax       = 4.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    
if(ptype==3):
    
    dx         = 20
    dz         = 20
    vmin       = 1.5
    vmax       = 5.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
            
if(ptype==4):
    
    dx         = 32
    dz         = 32
    vmin       = 1.5
    vmax       = 8.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])

if(ptype==5):
    
    dx         = 10
    dz         = 10
    vmin       = 1.5
    vmax       = 4.0
    vdt        = 2*np.array([0.06,0.1,0.2,0.3,0.4,0.5])

if(ptype==6):
    
    dx         = 5
    dz         = 5
    vmin       = 1.5
    vmax       = 4.0
    vdt        = 4*np.array([0.06,0.1,0.2,0.3,0.4,0.5])

if(ptype==7):
    
    dx         = 15
    dz         = 15
    vmin       = 1.5
    vmax       = 3.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])

if(ptype==8):
    
    dx         = 10
    dz         = 10
    vmin       = 1.5
    vmax       = 3.0
    vdt        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])

if(ptype==9):
    
    dx         = 10
    dz         = 10
    vmin       = 1.5
    vmax       = 3.0
    vdt        = np.array([0.5,1.0,1.5,2.0])

print('')
#==============================================================================

#==============================================================================
# Looping Velocity
#==============================================================================
ntimes   = len(vdt) 
nschemes = 9
vnorder  = 4
vvel     = np.linspace(vmin,vmax,vnorder+1)
mcoefs   = np.zeros((ntimes,nschemes,vnorder+1,nordem+1,nordem+1))

for dtype in range(0,ntimes):

    for i in range(0,vnorder+1):
    
        vmaxloc = vvel[i]
        rval    = (vdt[dtype]*vmaxloc)/min(dx,dz)
#==============================================================================

#==============================================================================
# Basic Weights
#==============================================================================
        T0                    = bw.calccoef(nordem)
        mcoefs[dtype,0,i,:,:] = T0[:,:]
    
        if(textprint==1):

            print('FD Scheme = Classical - Order =  %d - vmax = %f'%(nordem,vmaxloc))
            print(np.round(T0,nround))
            print('')
#==============================================================================

#==============================================================================
# T4 Type - Coef Yang Liu - A nearly globally optimal FD scheme based on 2D
# time-space-domain dispersion relation
#==============================================================================
        bint                  = 2.74
        tetaint               = np.pi/4
        T4                    = cotm14.calccoef(mvalue,bint,tetaint,rval)
        mshow                 = np.zeros((nordem+1,nordem+1))
        nmeio                 = int(nordem/2)
        mshow[nmeio,:]        = T4[:]
        mshow[:,nmeio]        = T4[:]
        mshow[nmeio,nmeio]    = 2*T4[nmeio]
        mcoefs[dtype,1,i,:,:] = mshow[:,:]
    
        if(textprint==1):
    
            print('FD Scheme = Cross2009_TE - Order =  %d - vmax = %f'%(nordem,vmaxloc))
            print(np.round(mshow,nround))
            print('')
#==============================================================================

#==============================================================================
# T7 Type - Coef Liu and Sen - Time–space domain high-order FD method for 2D 
# acoustic  wave equations
#==============================================================================
        tetaval               = np.pi/8
        T7                    = cotm32.calccoef(nordem,rval,tetaval)
        mshow                 = np.zeros((nordem+1,nordem+1))
        nmeio                 = int(nordem/2)
        mshow[nmeio,:]        = T7[:]
        mshow[:,nmeio]        = T7[:]
        mshow[nmeio,nmeio]    = 2*T7[nmeio]
        mcoefs[dtype,2,i,:,:] = mshow[:,:]    
    
        if(textprint==1):
    
            print('FD Scheme = Cross2013_LS - Order =  %d - vmax = %f'%(nordem,vmaxloc))
            print(np.round(mshow,nround))
            print('')
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
        nvalue                 = 1    
        T121                  = cotm71.calccoef(mvalue,nvalue,rval,rval) 
        mcoefs[dtype,3,i,:,:] = T121[:,:]
    
        if(textprint==1):
    
            print('FD Scheme = Cross_Rhombhus_2016_TE - Order =  %d - M = %d - N = %d - vmax = %f'%(2*mvalue,mvalue,nvalue,vmaxloc))
            print(np.round(T121,nround))
            print('')
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================    
        nvalue                = 1    
        T122                  = cotm72.calccoef(mvalue,nvalue,rval,rval)
        mcoefs[dtype,4,i,:,:] = T122[:,:]
        
        if(textprint==1):
    
            print('FD Scheme = Cross_Rhombhus_2016_LS - Order =  %d - M = %d - N = %d - vmax = %f'%(2*mvalue,mvalue,nvalue,vmaxloc))
            print(np.round(T122,nround))
            print('')
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
        nvalue                = int(mvalue/2)+1    
        T121                  = cotm71.calccoef(mvalue,nvalue,rval,rval) 
        mcoefs[dtype,5,i,:,:] = T121[:,:]
        
        if(textprint==1):
    
            print('FD Scheme = Cross_Rhombhus_2016_TE - Order =  %d - M = %d - N = %d - vmax = %f'%(2*mvalue,mvalue,nvalue,vmaxloc))
            print(np.round(T121,nround))
            print('')
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================    
        nvalue                = int(mvalue/2)+1    
        T122                  = cotm72.calccoef(mvalue,nvalue,rval,rval)
        mcoefs[dtype,6,i,:,:] = T122[:,:]
    
        if(textprint==1):
    
            print('FD Scheme = Cross_Rhombhus_2016_LS - Order =  %d - M = %d - N = %d - vmax = %f'%(2*mvalue,mvalue,nvalue,vmaxloc))
            print(np.round(T122,nround))
            print('')
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
        nvalue                = mvalue    
        T121                  = cotm71.calccoef(mvalue,nvalue,rval,rval) 
        mcoefs[dtype,7,i,:,:] = T121[:,:]
    
        if(textprint==1):
    
            print('FD Scheme = Cross_Rhombhus_2016_TE - Order =  %d - M = %d - N = %d - vmax = %f'%(2*mvalue,mvalue,nvalue,vmaxloc))
            print(np.round(T121,nround))
            print('')
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================    
        nvalue                = mvalue
        T122                  = cotm72.calccoef(mvalue,nvalue,rval,rval)
        mcoefs[dtype,8,i,:,:] = T122[:,:]
    
        if(textprint==1):
    
            print('FD Scheme = Cross_Rhombhus_2016_LS - Order =  %d - M = %d - N = %d - vmax = %f'%(2*mvalue,mvalue,nvalue,vmaxloc))
            print(np.round(T122,nround))
            print('')
#==============================================================================

#==============================================================================
# Velocity/Weights Interpolation
#==============================================================================
npoints = 10
vrange  = np.linspace(vmin,vmax,npoints)            
vresult = np.zeros((ntimes,nschemes,npoints))

for m2 in range(0,ntimes):

    for m1 in range(0,nschemes):

        if(textprint==1):

            print('Velocity Stencil: ')
            print(vvel)
            print('')

        npx       = mcoefs.shape[3]
        npy       = mcoefs.shape[4]
        npxm      = int(npx/2)
        npym      = int(npy/2)
        initialx  = -npxm
        initialy  =  npym
        pdeaux    = 0
        contcoef  = 0 

        for i in range(0,npx):
     
            for j in range(0,npy):
                             
                a   = int(initialx)
                b   = int(initialy)
                pxs = a
                pys = b                
        
                vel              = sym.Symbol('vel')
                
                list_points = list()
                
                for m3 in range(0,vvel.shape[0]):
                    
                    pair = (vvel[m3],mcoefs[m2,m1,m3,i,j])
                    list_points.append(pair)
                                
                #weight_interp    = interpolate(mcoefs[m2,m1,:,i,j],vel)
                weight_interp    = interpolate(list_points,vel)
                
                initialx         = initialx + 1
        
                if(textprint==1 and weight_interp!=0):
        
                    if(pxs==0): 
                
                        print('xpositon = x')        
            
                    elif(pxs>0):  
                
                        print('xpositon = x + %d'%pxs)
            
                    else:
                    
                        print('xpositon = x - %d'%np.abs(pxs))
                
                    if(pys==0): 
                
                        print('ypositon = y')
                
                    elif(pys>0):  
                    
                        print('ypositon = y + %d'%pys)
            
                    else:
                
                        print('ypositon = y - %d'%np.abs(pys))
        
                    print('Velocity Polinomial Interpolation: ')
                    print(weight_interp)
                    print('')

                if(a==0 and b==0): 
                  
                    for k in range(0,npoints):
            
                        velloc        = vrange[k]
                        eps           = 10**(-8)
                        val1          = interpolate(list_points,velloc+eps)
                        #val1          = interpolate(mcoefs[m2,m1,:,i,j],velloc+eps)
                        vresult[m2,m1,k] = val1
        
            initialx = -npxm
            initialy =  initialy - 1
#==============================================================================

#==============================================================================
# Graphical Construction
#==============================================================================
vticks  = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
vline   = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
vcolors = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']
vnames  = ['Classic', 'Cross2009', 'Cross2013', 'Cross2016_TE', 'Cross2016_LS', 'Cross_Rb2016_TE', 
          'Cross_Rb2016_LS', 'Rhombus2016_TE', 'Rhombus2016_LS']

plot.figure(figsize = (12,12))
plot.suptitle('Central Coefficient $a_{00}$ - dx = %.3f m - dz = %.3f m'%(dx,dz))
grid                = plot.GridSpec(2,2,wspace=0.1,hspace=0.5)
position_plot_listx = np.array([0,0,1,1])
position_plot_listy = np.array([0,1,0,1])

ntimes = len(vdt)

for k in range(0,ntimes):
    
    xpos     = int(position_plot_listx[k])
    ypos     = int(position_plot_listy[k])

    plot.subplot(grid[xpos,ypos])

    for i in range(0,nschemes):

        plot.plot(vrange,vresult[k,i,:],color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])

    max_valuey = np.amax(vresult)
    min_valuey = np.amin(vresult)

    max_valuex = np.amax(vrange)
    min_valuex = np.amin(vrange)

    plot.grid()
    plot.title('dt = %.3f ms'%vdt[k])
    if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
    plot.xticks(vrange)
    plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plot.ylim((min_valuey,max_valuey))
    ax = plot.gca()

    if(xpos==0 and ypos==0):
        
        ax.axes.xaxis.set_ticklabels([])
        plot.ylabel('$a_{00}$ Coefficient Value [ ]')
        ax.set_xticks(np.round(np.linspace(min_valuex, max_valuex, 5), 2))
        ax.set_yticks(np.round(np.linspace(min_valuey, max_valuey, 5), 2))

    if(xpos==0 and ypos==1):
        
        ax.set_yticks([],minor=True)
        ax.yaxis.set_ticklabels([])
        plot.minorticks_off()
        ax.axes.xaxis.set_ticklabels([])
        ax.set_xticks(np.round(np.linspace(min_valuex, max_valuex, 5), 2))
        ax.set_yticks(np.round(np.linspace(min_valuey, max_valuey, 5), 2))

    if(xpos==1 and ypos==0):
        
        plot.xlabel('Velocity [km/s]')
        plot.ylabel('$a_{00}$ Coefficient Value [ ]')
        ax.set_xticks(np.round(np.linspace(min_valuex, max_valuex, 5), 2))
        ax.set_yticks(np.round(np.linspace(min_valuey, max_valuey, 5), 2))

    if(xpos==1 and ypos==1):
     
        ax.set_yticks([],minor=True)
        ax.yaxis.set_ticklabels([])
        plot.minorticks_off()
        plot.xlabel('Velocity [km/s]')
        ax.set_xticks(np.round(np.linspace(min_valuex, max_valuex, 5), 2))
        ax.set_yticks(np.round(np.linspace(min_valuey, max_valuey, 5), 2))
    
plot.show()
plot.savefig('central_weight_%d.png'%(ptype),dpi=200,bbox_inches='tight')
plot.close()
#==============================================================================