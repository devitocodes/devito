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
import matplotlib.ticker       as mticker    
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   matplotlib              import ticker
from   numpy                   import linalg as la
from   matplotlib              import cm
#==============================================================================

#==============================================================================
# Configurações de Plot
#==============================================================================
plot.rc('font'  , size      = 12)     # controls default text sizes
plot.rc('axes'  , titlesize = 12)     # fontsize of the axes title
plot.rc('axes'  , labelsize = 12)     # fontsize of the x and y labels
plot.rc('xtick' , labelsize = 12)     # fontsize of the tick labels
plot.rc('ytick' , labelsize = 12)     # fontsize of the tick labels
plot.rc('legend', fontsize  = 12)     # legend fontsize
plot.rc('figure', titlesize = 12)     # fontsize of the figure title
#==============================================================================

#==============================================================================
plot.close("all")
#==============================================================================

#==============================================================================
# Testes de Leitura de Dados
#==============================================================================
ptype    = 9
normtype = 2

if(ptype==1): 
        
    nptx        = 101 
    npty        = 101    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])
    timevalue   = 450
    setup1      = ('data_save/teste1/dt1/',10000,3 ,1)
    setup2      = ('data_save/teste1/dt2/',6000 ,5 ,2)
    setup3      = ('data_save/teste1/dt3/',3000 ,10,3)
    setup4      = ('data_save/teste1/dt4/',2000 ,15,4)
    setup5      = ('data_save/teste1/dt5/',1500 ,20,5)
    setup6      = ('data_save/teste1/dt6/',1200 ,25,6)
    figsave     = 'figures/teste1/'
    vdts        = np.array([0.3,0.5,1.0,1.5,2.0,2.5])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,5]) 
    domain_setup = (0,2000,0,2000,0,3000)
    
if(ptype==2): 
    
    nptx        = 251 
    npty        = 251    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])   
    timevalue   = 450
    setup1      = ('data_save/teste2/dt1/',6000,5 ,1)
    setup2      = ('data_save/teste2/dt2/',3000,10,2)
    setup3      = ('data_save/teste2/dt3/',2000,15,3)
    setup4      = ('data_save/teste2/dt4/',1500,20,4)
    setup5      = ('data_save/teste2/dt5/',1200,25,5)
    setup6      = ('data_save/teste2/dt6/',1000,30,6)
    figsave     = 'figures/teste2/'
    vdts        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,5]) 
    domain_setup = (0,2500,0,2500,0,3000)

if(ptype==3): 
    
    nptx        = 601 
    npty        = 201    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([4000.0,4000.0,4000.0,6000.0,6000.0,6000.0,8000.0,8000.0,8000.0])   
    ypositionv  = np.array([2000.0,2500.0,3000.0,2000.0,2500.0,3000.0,2000.0,2500.0,3000.0])    
    timevalue   = 1800
    setup1      = ('data_save/teste3/dt1/',6000,5 ,1)
    setup2      = ('data_save/teste3/dt2/',3000,10,2)
    setup3      = ('data_save/teste3/dt3/',2000,15,3)
    setup4      = ('data_save/teste3/dt4/',1500,20,4)
    setup5      = ('data_save/teste3/dt5/',1200,25,5)
    setup6      = ('data_save/teste3/dt6/',1000,30,6)
    figsave     = 'figures/teste3/'
    vdts        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,5]) 
    domain_setup = (0,12000,0,4000,0,3000)

if(ptype==5): 
        
    nptx        = 201 
    npty        = 201    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])
    timevalue   = 450
    setup1      = ('data_save/teste5/dt1/',10000,3 ,1)
    setup2      = ('data_save/teste5/dt2/',6000 ,5 ,2)
    setup3      = ('data_save/teste5/dt3/',3000 ,10,3)
    setup4      = ('data_save/teste5/dt4/',2000 ,15,4)
    setup5      = ('data_save/teste5/dt5/',1500 ,20,5)
    setup6      = ('data_save/teste5/dt6/',1200 ,25,6)
    figsave     = 'figures/teste5/'
    vdts        = np.array([0.3,0.5,1.0,1.5,2.0,2.5])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,5]) 
    domain_setup = (0,2000,0,2000,0,3000)

if(ptype==6): 
        
    nptx        = 401 
    npty        = 401    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])
    timevalue   = 450
    setup1      = ('data_save/teste6/dt1/',10000,3 ,1)
    setup2      = ('data_save/teste6/dt2/',6000 ,5 ,2)
    setup3      = ('data_save/teste6/dt3/',3000 ,10,3)
    setup4      = ('data_save/teste6/dt4/',2000 ,15,4)
    setup5      = ('data_save/teste6/dt5/',1500 ,20,5)
    setup6      = ('data_save/teste6/dt6/',1200 ,25,6)
    figsave     = 'figures/teste6/'
    vdts        = np.array([0.3,0.5,1.0,1.5,2.0,2.5])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,5]) 
    domain_setup = (0,2000,0,2000,0,3000)

if(ptype==7): 
        
    nptx        = 201 
    npty        = 201    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([750.0,2250.0, 750.0,2250.0])
    ypositionv  = np.array([750.0, 750.0,2250.0,2250.0])
    timevalue   = 900
    setup1      = ('data_save/teste7/dt1/',6000,5 ,1)
    setup2      = ('data_save/teste7/dt2/',3000,10,2)
    setup3      = ('data_save/teste7/dt3/',2000,15,3)
    setup4      = ('data_save/teste7/dt4/',1500,20,4)
    setup5      = ('data_save/teste7/dt5/',1200,25,5)
    setup6      = ('data_save/teste7/dt6/',1000,30,6)
    figsave     = 'figures/teste7/'
    vdts        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,5]) 
    domain_setup = (0,3000,0,3000,0,3000)
    
if(ptype==8): 
        
    nptx        = 201 
    npty        = 201    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])
    timevalue   = 540
    setup1      = ('data_save/teste8/dt1/',6000,5 ,1)
    setup2      = ('data_save/teste8/dt2/',3000,10,2)
    setup3      = ('data_save/teste8/dt3/',2000,15,3)
    setup4      = ('data_save/teste8/dt4/',1500,20,4)
    setup5      = ('data_save/teste8/dt5/',1200,25,5)
    setup6      = ('data_save/teste8/dt6/',1000,30,6)
    figsave     = 'figures/teste8/'
    vdts        = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,5]) 
    domain_setup = (0,2000,0,2000,0,3000)

if(ptype==9): 
        
    nptx        = 201 
    npty        = 201    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])
    timevalue   = 3000
    setup1      = ('data_save/teste9/dt1/',6000,5 ,1)
    setup2      = ('data_save/teste9/dt2/',3000,10,2)
    setup3      = ('data_save/teste9/dt3/',2000,15,3)
    setup4      = ('data_save/teste9/dt4/',1500,20,4)
    setup5      = ('data_save/teste9/dt5/',1200,25,5)
    setup6      = ('data_save/teste9/dt6/',1200,25,6)
    figsave     = 'figures/teste9/'
    vdts        = np.array([0.5,1.0,1.5,2.0,2.5,2.5])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,5]) 
    domain_setup = (0,2000,0,2000,0,3000)
#==============================================================================

#==============================================================================
# Vetores de Configurações
#==============================================================================
vmethod0 = np.array([0,1,1,1,1,1,1,1,1])
vmethod1 = np.array([1,1,3,7,7,7,7,7,7])
vmethod2 = np.array([1,4,2,1,2,1,2,1,2])
vmethod3 = np.array([2,4,6,8,10,12,14,16,18,20])
nteste   = vmethod3.shape[0]

l1   = np.zeros(nteste)
l2   = np.zeros(nteste)
l3   = np.zeros(nteste)
l4   = np.zeros(nteste)
l5   = np.zeros(nteste)
l6   = np.zeros(nteste)
l7   = np.zeros(nteste)
l8   = np.zeros(nteste)
l9   = np.zeros(nteste)

for i in range(0,nteste):
    
    l1[i]  = 1
    l2[i]  = 1
    l3[i]  = 1
    l4[i]  = 1
    l5[i]  = 1
    l6[i]  = int(vmethod3[i]/2)
    l7[i]  = int(vmethod3[i]/2)
    l8[i]  = int(vmethod3[i]/4 + 1)
    l9[i]  = int(vmethod3[i]/4 + 1)

vmethod4 = [l1,l2,l3,l4,l5,l6,l7,l8,l9]

total_configs = 0
list_config   = []

for i in range(0,vmethod0.shape[0]):
    
    scheme  = i
    peso    = vmethod0[i]
    wauthor = vmethod1[i]
    wtype   = vmethod2[i]
    vnvalue = vmethod4[i]
    
    for l in range(0,vmethod3.shape[0]):
        
        mvalue = vmethod3[l]
        nvalue = int(vnvalue[l])
        config = (peso,wauthor,wtype,mvalue,nvalue,scheme)                    
        total_configs = total_configs + 1
        list_config.append(config)

list_config = list(set(list_config))
nconfig     = len(list_config)
#==============================================================================

#==============================================================================
# Carregando Soluções
#==============================================================================
setup                = setup1
rec_num1             = np.load('%srec_num_%d.npy'%(setup[0],setup[3]))
solplot1             = np.load('%ssolplot_%d.npy'%(setup[0],setup[3]))
rec_select_num1      = np.load('%srec_select_num_%d.npy'%(setup[0],setup[3]))
solplot_select1      = np.load('%ssolplot_select_%d.npy'%(setup[0],setup[3]))
solplot_ref1         = np.load('%ssolplot_ref_%d.npy'%(setup[0],setup[3]))
rec_ref1             = np.load('%srec_ref_%d.npy'%(setup[0],setup[3]))
rec_select_ref1      = np.load('%srec_select_ref_%d.npy'%(setup[0],setup[3]))
timev_disp1          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec1           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv1             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup2
rec_num2             = np.load('%srec_num_%d.npy'%(setup[0],setup[3]))
solplot2             = np.load('%ssolplot_%d.npy'%(setup[0],setup[3]))
rec_select_num2      = np.load('%srec_select_num_%d.npy'%(setup[0],setup[3]))
solplot_select2      = np.load('%ssolplot_select_%d.npy'%(setup[0],setup[3]))
solplot_ref2         = np.load('%ssolplot_ref_%d.npy'%(setup[0],setup[3]))
rec_ref2             = np.load('%srec_ref_%d.npy'%(setup[0],setup[3]))
rec_select_ref2      = np.load('%srec_select_ref_%d.npy'%(setup[0],setup[3]))
timev_disp2          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec2           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv2             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup3
rec_num3             = np.load('%srec_num_%d.npy'%(setup[0],setup[3]))
solplot3             = np.load('%ssolplot_%d.npy'%(setup[0],setup[3]))
rec_select_num3      = np.load('%srec_select_num_%d.npy'%(setup[0],setup[3]))
solplot_select3      = np.load('%ssolplot_select_%d.npy'%(setup[0],setup[3]))
solplot_ref3         = np.load('%ssolplot_ref_%d.npy'%(setup[0],setup[3]))
rec_ref3             = np.load('%srec_ref_%d.npy'%(setup[0],setup[3]))
rec_select_ref3      = np.load('%srec_select_ref_%d.npy'%(setup[0],setup[3]))
timev_disp3          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec3           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv3             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup4
rec_num4             = np.load('%srec_num_%d.npy'%(setup[0],setup[3]))
solplot4             = np.load('%ssolplot_%d.npy'%(setup[0],setup[3]))
rec_select_num4      = np.load('%srec_select_num_%d.npy'%(setup[0],setup[3]))
solplot_select4      = np.load('%ssolplot_select_%d.npy'%(setup[0],setup[3]))
solplot_ref4         = np.load('%ssolplot_ref_%d.npy'%(setup[0],setup[3]))
rec_ref4             = np.load('%srec_ref_%d.npy'%(setup[0],setup[3]))
rec_select_ref4      = np.load('%srec_select_ref_%d.npy'%(setup[0],setup[3]))
timev_disp4          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec4           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv4             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup5
rec_num5             = np.load('%srec_num_%d.npy'%(setup[0],setup[3]))
solplot5             = np.load('%ssolplot_%d.npy'%(setup[0],setup[3]))
rec_select_num5      = np.load('%srec_select_num_%d.npy'%(setup[0],setup[3]))
solplot_select5      = np.load('%ssolplot_select_%d.npy'%(setup[0],setup[3]))
solplot_ref5         = np.load('%ssolplot_ref_%d.npy'%(setup[0],setup[3]))
rec_ref5             = np.load('%srec_ref_%d.npy'%(setup[0],setup[3]))
rec_select_ref5      = np.load('%srec_select_ref_%d.npy'%(setup[0],setup[3]))
timev_disp5          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec5           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv5             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup6
rec_num6             = np.load('%srec_num_%d.npy'%(setup[0],setup[3]))
solplot6             = np.load('%ssolplot_%d.npy'%(setup[0],setup[3]))
rec_select_num6      = np.load('%srec_select_num_%d.npy'%(setup[0],setup[3]))
solplot_select6      = np.load('%ssolplot_select_%d.npy'%(setup[0],setup[3]))
solplot_ref6         = np.load('%ssolplot_ref_%d.npy'%(setup[0],setup[3]))
rec_ref6             = np.load('%srec_ref_%d.npy'%(setup[0],setup[3]))
rec_select_ref6      = np.load('%srec_select_ref_%d.npy'%(setup[0],setup[3]))
timev_disp6          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec6           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv6             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

rec_num                            = np.zeros((6,rec_num1.shape[0],rec_num1.shape[1],rec_num1.shape[2]))
rec_num[0,:,0:rec_num1.shape[1],:] = rec_num1[:,:,:]
rec_num[1,:,0:rec_num2.shape[1],:] = rec_num2[:,:,:]
rec_num[2,:,0:rec_num3.shape[1],:] = rec_num3[:,:,:]
rec_num[3,:,0:rec_num4.shape[1],:] = rec_num4[:,:,:]
rec_num[4,:,0:rec_num5.shape[1],:] = rec_num5[:,:,:]
rec_num[5,:,0:rec_num6.shape[1],:] = rec_num6[:,:,:]

solplot            = np.zeros((6,solplot1.shape[0],solplot1.shape[1],solplot1.shape[2],solplot1.shape[3]))
solplot[0,:,:,:,:] = solplot1[:,:,:,:]
solplot[1,:,:,:,:] = solplot2[:,:,:,:]
solplot[2,:,:,:,:] = solplot3[:,:,:,:]
solplot[3,:,:,:,:] = solplot4[:,:,:,:]
solplot[4,:,:,:,:] = solplot5[:,:,:,:]
solplot[5,:,:,:,:] = solplot6[:,:,:,:]

rec_select_num                                   = np.zeros((6,rec_select_num1.shape[0],rec_select_num1.shape[1],rec_select_num1.shape[2]))
rec_select_num[0,:,0:rec_select_num1.shape[1],:] = rec_select_num1[:,:,:]
rec_select_num[1,:,0:rec_select_num2.shape[1],:] = rec_select_num2[:,:,:]
rec_select_num[2,:,0:rec_select_num3.shape[1],:] = rec_select_num3[:,:,:]
rec_select_num[3,:,0:rec_select_num4.shape[1],:] = rec_select_num4[:,:,:]
rec_select_num[4,:,0:rec_select_num5.shape[1],:] = rec_select_num5[:,:,:]
rec_select_num[5,:,0:rec_select_num6.shape[1],:] = rec_select_num6[:,:,:]
 
solplot_select                                   = np.zeros((6,solplot_select1.shape[0],solplot_select1.shape[1],solplot_select1.shape[2]))
solplot_select[0,:,:,0:solplot_select1.shape[2]] = solplot_select1[:,:,:]
solplot_select[1,:,:,0:solplot_select2.shape[2]] = solplot_select2[:,:,:]
solplot_select[2,:,:,0:solplot_select3.shape[2]] = solplot_select3[:,:,:]
solplot_select[3,:,:,0:solplot_select4.shape[2]] = solplot_select4[:,:,:]
solplot_select[4,:,:,0:solplot_select5.shape[2]] = solplot_select5[:,:,:]
solplot_select[5,:,:,0:solplot_select6.shape[2]] = solplot_select6[:,:,:]

solplot_ref          = np.zeros((6,solplot_ref1.shape[0],solplot_ref1.shape[1],solplot_ref1.shape[2]))
solplot_ref[0,:,:,:] = solplot_ref1[:,:,:]
solplot_ref[1,:,:,:] = solplot_ref2[:,:,:]
solplot_ref[2,:,:,:] = solplot_ref3[:,:,:]
solplot_ref[3,:,:,:] = solplot_ref4[:,:,:]
solplot_ref[4,:,:,:] = solplot_ref5[:,:,:]
solplot_ref[5,:,:,:] = solplot_ref6[:,:,:]

rec_ref                          = np.zeros((6,rec_ref1.shape[0],rec_ref1.shape[1]))
rec_ref[0,0:rec_ref1.shape[0],:] = rec_ref1[:,:]
rec_ref[1,0:rec_ref2.shape[0],:] = rec_ref2[:,:]
rec_ref[2,0:rec_ref3.shape[0],:] = rec_ref3[:,:]
rec_ref[3,0:rec_ref4.shape[0],:] = rec_ref4[:,:]
rec_ref[4,0:rec_ref5.shape[0],:] = rec_ref5[:,:]
rec_ref[5,0:rec_ref6.shape[0],:] = rec_ref6[:,:]

rec_select_ref                                 = np.zeros((6,rec_select_ref1.shape[0],rec_select_ref1.shape[1]))
rec_select_ref[0,0:rec_select_ref1.shape[0],:] = rec_select_ref1[:,:]
rec_select_ref[1,0:rec_select_ref2.shape[0],:] = rec_select_ref2[:,:]
rec_select_ref[2,0:rec_select_ref3.shape[0],:] = rec_select_ref3[:,:]
rec_select_ref[3,0:rec_select_ref4.shape[0],:] = rec_select_ref4[:,:]
rec_select_ref[4,0:rec_select_ref5.shape[0],:] = rec_select_ref5[:,:]
rec_select_ref[5,0:rec_select_ref6.shape[0],:] = rec_select_ref6[:,:]
 
timev_disp      = np.zeros((6,timev_disp1.shape[0]))
timev_disp[0,:] = timev_disp1[:]
timev_disp[1,:] = timev_disp2[:]
timev_disp[2,:] = timev_disp3[:]
timev_disp[3,:] = timev_disp4[:]
timev_disp[4,:] = timev_disp5[:]
timev_disp[5,:] = timev_disp6[:]

timev_rec                          = np.zeros((6,timev_rec1.shape[0]))
timev_rec[0,0:timev_rec1.shape[0]] = timev_rec1[:]
timev_rec[1,0:timev_rec2.shape[0]] = timev_rec2[:]
timev_rec[2,0:timev_rec3.shape[0]] = timev_rec3[:]
timev_rec[3,0:timev_rec4.shape[0]] = timev_rec4[:]
timev_rec[4,0:timev_rec5.shape[0]] = timev_rec5[:]
timev_rec[5,0:timev_rec6.shape[0]] = timev_rec6[:]

ordersv      = np.zeros((6,ordersv1.shape[0]))
ordersv[0,:] = ordersv1[:]
ordersv[1,:] = ordersv2[:]
ordersv[2,:] = ordersv3[:]
ordersv[3,:] = ordersv4[:]
ordersv[4,:] = ordersv5[:]
ordersv[5,:] = ordersv6[:]

vnames = ['Classic', 'Cross2009', 'Cross2013', 'Cross2016_TE', 'Cross2016_LS', 'Cross_Rb2016_TE', 
          'Cross_Rb2016_LS', 'Rhombus2016_TE', 'Rhombus2016_LS']

vdts_select  = np.zeros(times_cho.shape[0])

for k in range(0,times_cho.shape[0]):
        
    vdts_select[k] = vdts[int(times_cho[k])]
#==============================================================================

#==============================================================================
# Manipulando Solução de Referência
#==============================================================================
timepos    = np.zeros(6)
timeposrec = np.zeros(6)

for i in range(0,6):

    for j in range(0,solplot.shape[2]):

        if(timevalue==timev_disp[i][j]): timepos[i] = j
    
    for j in range(0,rec_select_num.shape[2]):

        if(timevalue==timev_rec[i][j]): timeposrec[i] = j
#==============================================================================

#==============================================================================
list0 = []
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []

for i in range(0,nconfig):

    config = list_config[i]
    
    position = i
    peso    = config[0]
    wauthor = config[1]
    wtype   = config[2]
    mvalue  = config[3]
    nvalue  = config[4]
    scheme  = config[5]
        
    pair = (peso,wauthor,wtype,mvalue,nvalue,position)
    
    if(scheme==0): list0.append(pair)
    if(scheme==1): list1.append(pair)
    if(scheme==2): list2.append(pair)
    if(scheme==3): list3.append(pair)
    if(scheme==4): list4.append(pair)
    if(scheme==5): list5.append(pair)
    if(scheme==6): list6.append(pair)
    if(scheme==7): list7.append(pair)
    if(scheme==8): list8.append(pair)
    
list0 = list(sorted(list0))
list1 = list(sorted(list1))
list2 = list(sorted(list2))
list3 = list(sorted(list3))
list4 = list(sorted(list4))
list5 = list(sorted(list5))
list6 = list(sorted(list6))
list7 = list(sorted(list7))
list8 = list(sorted(list8))

list_scheme = [list0,list1,list2,list3,list4,list5,list6,list7,list8]
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 7
#==============================================================================
def plot7(solplot_ref,solplot,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timepos):
    
    fscale    =  10**(-3)    

    position_plot_listx = np.array([0,0,1,1,2,2,3,3,4,4])
    position_plot_listy = np.array([0,1,0,1,0,1,0,1,0,1])
    ntimes   = vdts.shape[0]
    norders  = ordersv[0].shape[0]
    nschemes = len(list_scheme) + 1
    
    x0        = domain_setup[0]
    x1        = domain_setup[1]
    y0        = domain_setup[2]
    y1        = domain_setup[3]
    timevalue = fscale*timevalue    
    if(ptype==1 or ptype==2 or ptype==5 or ptype==6 or ptype==7 or ptype==8 or ptype==9): factor = 10
    if(ptype==3 or ptype==4): factor = 50
    scale     = np.amax(solplot_ref)/factor

    for k1 in range(0,ntimes):
        
        timeposloc = int(timepos[k1])
    
        for k2 in range(0,norders):

            fig1 = plot.figure(figsize = (5,12))
   
            plot.suptitle('Displacement - Space Order = %d \n T = %.2f s - dt = %.3f ms'%(ordersv[0][k2],timevalue,vdts[k1]),fontsize=10)
 
            grid = plot.GridSpec(5,2,wspace=0.45,hspace=0.3)
   
            for k3 in range(0,nschemes):
                
                if(k3==0):
                    
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    sol = solplot_ref[k1,timeposloc,:,:]
                    plot.title('Reference',fontsize=7)

                else:
                
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    listm      = list_scheme[k3-1]
                    index      = listm[k2][-1]
                    sol = solplot[k1,index,timeposloc,:,:]
                    plot.title(vnames[k3-1],fontsize=7)

                ax     = plot.gca()
                extent = [fscale*x0,fscale*x1, fscale*y1, fscale*y0]
                fig    = plot.imshow(np.transpose(sol),vmin=-scale,vmax=scale,cmap=cm.binary, extent=extent,interpolation='bicubic', aspect='auto')
                plot.grid()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.tick_params(axis="x", labelsize=6)
                ax.tick_params(axis="y", labelsize=6)

                if(ypos==0): 
                   
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
                
                if(ypos==1):

                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))

                if(xpos==4): 
                
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
                
                if(xpos==4 and ypos==0): 
                    
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))

            cb_ax = fig1.add_axes([0.001, 0.06, 1.0 , 0.02])
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar = fig1.colorbar(fig, cax=cb_ax,format='%.2e',orientation='horizontal')
            cbar.locator = tick_locator
            cbar.ax.tick_params(labelsize=6)
            cbar.update_ticks()
            plot.show()
            plot.savefig('%scomp_methods/snapshots/disp_order_%d_dt_%f_%d.png'%(figsave,ordersv[0][k2],vdts[k1],ptype),dpi=200,bbox_inches='tight')
            plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 8
#==============================================================================
def plot8(rec_ref,rec_num,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timeposrec):
    
    fscale    =  10**(-3)    

    position_plot_listx = np.array([0,0,1,1,2,2,3,3,4,4])
    position_plot_listy = np.array([0,1,0,1,0,1,0,1,0,1])
    ntimes   = vdts.shape[0]
    norders  = ordersv[0].shape[0]
    nschemes = len(list_scheme) + 1
    
    x0        = domain_setup[0]
    x1        = domain_setup[1]
    t0        = domain_setup[4]
    tn        = timevalue
    timevalue = fscale*timevalue
    if(ptype==1 or ptype==2 or ptype==5 or ptype==6 or ptype==7 or ptype==8 or ptype==9): factor = 10
    if(ptype==3 or ptype==4): factor = 50
    scale     = np.amax(rec_ref)/factor
    
    for k1 in range(0,ntimes):
        
        timeposrecloc = int(timeposrec[k1])
        
        for k2 in range(0,norders):
               
            fig1 = plot.figure(figsize = (5,12))
   
            plot.suptitle('Receiver - Space Order = %d \n T = %.2f s - dt = %.3f ms'%(ordersv[0][k2],timevalue,vdts[k1]),fontsize=10)
 
            grid = plot.GridSpec(5,2,wspace=0.45,hspace=0.3)
   
            for k3 in range(0,nschemes):
                
                if(k3==0):
                    
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    setup      = setup_list[k1]
                    posfinal   = timeposrecloc
                    rec = rec_ref[k1,0:posfinal,:]
                    plot.title('Reference',fontsize=7)
                    
                else:
                    
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    setup      = setup_list[k1]
                    posfinal   = timeposrecloc
                    listm  = list_scheme[k3-1]
                    index  = listm[k2][-1]
                    rec    = rec_num[k1,index,0:posfinal,:]
                    plot.title(vnames[k3-1],fontsize=7)
                
                ax = plot.gca()
                extent = [fscale*x0,fscale*x1, fscale*tn, fscale*t0]
                fig = plot.imshow(rec,vmin=-scale, vmax=scale,cmap=cm.binary, extent=extent,interpolation='bicubic', aspect='auto')
                plot.grid()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.tick_params(axis="x", labelsize=6)
                ax.tick_params(axis="y", labelsize=6)

                if(ypos==0): 
                   
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
                
                if(ypos==1):

                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))

                if(xpos==4): 
                
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
                    
                if(xpos==4 and ypos==0): 
            
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
                    ax.xaxis.set_major_locator(plot.MaxNLocator(4))

            cb_ax = fig1.add_axes([0.001, 0.06, 1.0 , 0.02])
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar = fig1.colorbar(fig, cax=cb_ax,format='%.2e',orientation='horizontal')
            cbar.locator = tick_locator
            cbar.ax.tick_params(labelsize=6)
            cbar.update_ticks()
            plot.show()
            plot.savefig('%scomp_methods/snapshots/rec_order_%d_dt_%f_%d.png'%(figsave,ordersv[0][k2],vdts[k1],ptype),dpi=200,bbox_inches='tight')
            plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 7
#==============================================================================
def plot9(solplot_ref,solplot,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timepos):
    
    fscale    =  10**(-3)    

    position_plot_listx = np.array([0,0,1,1,2,2,3,3,4,4])
    position_plot_listy = np.array([0,1,0,1,0,1,0,1,0,1])
    ntimes   = vdts.shape[0]
    norders  = ordersv[0].shape[0]
    nschemes = len(list_scheme) + 1
    
    x0        = domain_setup[0]
    x1        = domain_setup[1]
    y0        = domain_setup[2]
    y1        = domain_setup[3]
    timevalue = fscale*timevalue    
    if(ptype==1 or ptype==2 or ptype==5 or ptype==6 or ptype==7 or ptype==8 or ptype==9): factor = 10
    if(ptype==3 or ptype==4): factor = 50
    scale     = np.amax(solplot_ref)/factor

    for k1 in range(0,ntimes):
        
        timeposloc = int(timepos[k1])
    
        for k2 in range(0,norders):

            fig1 = plot.figure(figsize = (5,12))
   
            plot.suptitle('Displacement Error - Space Order = %d \n T = %.2f s - dt = %.3f ms'%(ordersv[0][k2],timevalue,vdts[k1]),fontsize=10)
 
            grid = plot.GridSpec(5,2,wspace=0.45,hspace=0.3)
   
            for k3 in range(0,nschemes):
                
                if(k3==0):
                    
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    sol = solplot_ref[k1,timeposloc,:,:]
                    plot.title('Reference',fontsize=7)

                else:
                
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    listm      = list_scheme[k3-1]
                    index      = listm[k2][-1]
                    sol = solplot_ref[k1,timeposloc,:,:]-solplot[k1,index,timeposloc,:,:]
                    norm = np.linalg.norm(np.reshape(sol,-1),2)
                    
                    if(norm>10**(4)):
                        
                        plot.title(vnames[k3-1]+' Norm2 = Instable',fontsize=7)
                    
                    else:
                        
                        plot.title(vnames[k3-1]+' Norm2 = %.3f'%norm,fontsize=7)


                ax     = plot.gca()
                extent = [fscale*x0,fscale*x1, fscale*y1, fscale*y0]
                fig    = plot.imshow(np.transpose(sol),vmin=-scale,vmax=scale,cmap=cm.binary, extent=extent,interpolation='bicubic', aspect='auto')
                plot.grid()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.tick_params(axis="x", labelsize=6)
                ax.tick_params(axis="y", labelsize=6)

                if(ypos==0): 
                   
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
                
                if(ypos==1):

                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))

                if(xpos==4): 
                
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
                
                if(xpos==4 and ypos==0): 
                    
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))

            cb_ax = fig1.add_axes([0.001, 0.06, 1.0 , 0.02])
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar = fig1.colorbar(fig, cax=cb_ax,format='%.2e',orientation='horizontal')
            cbar.locator = tick_locator
            cbar.ax.tick_params(labelsize=6)
            cbar.update_ticks()
            plot.show()
            plot.savefig('%scomp_methods/errors/disp_order_error_%d_dt_%f_%d.png'%(figsave,ordersv[0][k2],vdts[k1],ptype),dpi=200,bbox_inches='tight')
            plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 8
#==============================================================================
def plot10(rec_ref,rec_num,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timeposrec):
    
    fscale    =  10**(-3)    

    position_plot_listx = np.array([0,0,1,1,2,2,3,3,4,4])
    position_plot_listy = np.array([0,1,0,1,0,1,0,1,0,1])
    ntimes   = vdts.shape[0]
    norders  = ordersv[0].shape[0]
    nschemes = len(list_scheme) + 1
    
    x0        = domain_setup[0]
    x1        = domain_setup[1]
    t0        = domain_setup[4]
    tn        = timevalue
    timevalue = fscale*timevalue
    if(ptype==1 or ptype==2 or ptype==5 or ptype==6 or ptype==7 or ptype==8 or ptype==9): factor = 10
    if(ptype==3 or ptype==4): factor = 50
    scale     = np.amax(rec_ref)/factor
    
    for k1 in range(0,ntimes):
        
        timeposrecloc = int(timeposrec[k1])
        
        for k2 in range(0,norders):
               
            fig1 = plot.figure(figsize = (5,12))
   
            plot.suptitle('Receiver Error - Space Order = %d \n T = %.2f s - dt = %.3f ms'%(ordersv[0][k2],timevalue,vdts[k1]),fontsize=10)
 
            grid = plot.GridSpec(5,2,wspace=0.45,hspace=0.3)
   
            for k3 in range(0,nschemes):
                
                if(k3==0):
                    
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    setup      = setup_list[k1]
                    posfinal   = timeposrecloc
                    rec = rec_ref[k1,0:posfinal,:]
                    plot.title('Reference',fontsize=7)
                    
                else:
                    
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    setup      = setup_list[k1]
                    posfinal   = timeposrecloc
                    listm  = list_scheme[k3-1]
                    index  = listm[k2][-1]
                    rec    = rec_ref[k1,0:posfinal,:]- rec_num[k1,index,0:posfinal,:]                    
                    norm = np.linalg.norm(np.reshape(rec,-1),2)
                    
                    if(norm>10**(4)):
                        
                        plot.title(vnames[k3-1]+' Norm2 = Instable',fontsize=7)
                    
                    else:
                        
                        plot.title(vnames[k3-1]+' Norm2 = %.3f'%norm,fontsize=7)
                
                ax = plot.gca()
                extent = [fscale*x0,fscale*x1, fscale*tn, fscale*t0]
                fig = plot.imshow(rec,vmin=-scale, vmax=scale,cmap=cm.binary, extent=extent,interpolation='bicubic', aspect='auto')
                plot.grid()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.tick_params(axis="x", labelsize=6)
                ax.tick_params(axis="y", labelsize=6)

                if(ypos==0): 
                   
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
                
                if(ypos==1):

                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))

                if(xpos==4): 
                
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
                    
                if(xpos==4 and ypos==0): 
            
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
                    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
                    ax.xaxis.set_major_locator(plot.MaxNLocator(4))

            cb_ax = fig1.add_axes([0.001, 0.06, 1.0 , 0.02])
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar = fig1.colorbar(fig, cax=cb_ax,format='%.2e',orientation='horizontal')
            cbar.locator = tick_locator
            cbar.ax.tick_params(labelsize=6)
            cbar.update_ticks()
            plot.show()
            plot.savefig('%scomp_methods/errors/rec_order_error_%d_dt_%f_%d.png'%(figsave,ordersv[0][k2],vdts[k1],ptype),dpi=200,bbox_inches='tight')
            plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados 
#==============================================================================
P7  = plot7(solplot_ref,solplot,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timepos)
P8  = plot8(rec_ref,rec_num,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timeposrec)
P9  = plot9(solplot_ref,solplot,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timepos)
P10 = plot10(rec_ref,rec_num,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timeposrec)
#==============================================================================
