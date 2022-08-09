#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as     np
import sys
from   scipy.integrate         import quad
import matplotlib.pyplot       as plot
import time                    as tm
#==============================================================================

#==============================================================================
# Shell Comands
#==============================================================================
import os
os.system('clear')
#==============================================================================

#==============================================================================
# Print Configurations
#==============================================================================
np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
#==============================================================================

#==============================================================================
# Basic Coefs
#==============================================================================
sys.path.insert(0, './basic_weights')
import basic_weights            as coefbasic
#==============================================================================

#==============================================================================
# Square Rhombus Adpt Coefs
#==============================================================================
sys.path.insert(0, './square_rhombus_adpt_weights')
import square_rhombus_adpt_weights   as coefsquarerhoadpt
#==============================================================================

#==============================================================================
plot.close("all")
print_allow = 1
nordem      = 8
#==============================================================================

#==============================================================================
# T4 - Square Adopt Rhombus Coefs
#==============================================================================
print('')
print('=========================================================================')
print('Square Rhombus Adaptative Scheme: ')
print('=========================================================================')
print('')

mvalue           = int(nordem/2)
exttrapmax       = int(0.5*(mvalue**2+mvalue))+1
exttrap          = np.random.randint(low=0,high=exttrapmax,size=1)[0]
hx               = 0.01
hz               = 0.01
hvalue           = min(hx,hz)
T2               = coefsquarerhoadpt.calccoef(mvalue,exttrap,hvalue)

if(print_allow==1):
        
    print('')
    print('=========================================================================')
    #print(hx**2*T2)
    print(hx**2*T2[T2!=0])    
    print('=========================================================================')
    print('')
#==============================================================================

#==============================================================================
# T1 - Basic Coefs
#==============================================================================
print('')
print('=========================================================================')
print('Classic Cross-Line Scheme: ')
print('=========================================================================')
print('')

T11, T12 = coefbasic.calccoef(nordem)

if(print_allow==1):
   
    print('')
    print('=========================================================================')
    print(T12[T12!=0])
    #print(T12)
    print('=========================================================================')
    print('')
#     G1 = plot.spy(T12,aspect='equal',markersize=5)
#==============================================================================

#==============================================================================
fig1 = plot.figure(figsize = (12,12))

grid = plot.GridSpec(1,2,wspace=0.4,hspace=0.2)
plot.suptitle('Stencil Schemes')

position_plot_listx = np.array([0,0])
position_plot_listy = np.array([0,1])

xpos = int(position_plot_listx[0])
ypos = int(position_plot_listy[0])    
plot.subplot(grid[xpos,ypos])
plot.spy(T2,aspect='equal',markersize=8)
plot.spy(T12,aspect='equal',markersize=8,color='red')
plot.title('Square-Rhombus Adaptative Scheme \n \n Cross-Points = %d and Extra Points = %d \n'%(mvalue,exttrap))
ax = plot.gca()
plot.grid()
vtikx = np.arange(-mvalue,mvalue+1) 
ax.set_xticks(np.arange(0,len(vtikx)))    
ax.axes.xaxis.set_ticklabels(vtikx)
ax.set_yticks(np.arange(0,len(vtikx)))
ax.axes.yaxis.set_ticklabels(np.flip(vtikx))
ax.xaxis.set_ticks_position('bottom')
plot.xlabel('[X Direction]')
plot.ylabel('[Z Direction]')

xpos = int(position_plot_listx[1])
ypos = int(position_plot_listy[1])    
plot.subplot(grid[xpos,ypos])
plot.spy(T12,aspect='equal',markersize=8,color='red')
plot.title('Cross-line Scheme \n \n Cross-Points = %d \n'%(mvalue))
ax = plot.gca()
plot.grid()
vtikx = np.arange(-mvalue,mvalue+1) 
ax.set_xticks(np.arange(0,len(vtikx)))    
ax.axes.xaxis.set_ticklabels(vtikx)
ax.set_yticks(np.arange(0,len(vtikx)))
ax.axes.yaxis.set_ticklabels(np.flip(vtikx))
ax.xaxis.set_ticks_position('bottom')
plot.xlabel('[X Direction]')
plot.ylabel('[Z Direction]')

plot.show()
#==============================================================================

#==============================================================================
len1 = len(T2[T2!=0])
len2 = len(T12[T12!=0])

if(len1==len2):

    max_dif = np.amax(np.abs(hx**2*T2[T2!=0]-T12[T12!=0]))

    print('')
    print('================================================')
    print('Maximum Difference of Coefficients: %.3e'%max_dif)
    print('================================================')
    print('')
#==============================================================================