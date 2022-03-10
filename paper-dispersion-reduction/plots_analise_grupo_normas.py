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
    vdts        = np.array([0.5,1.0,1.5])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,2]) 
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
    vdts        = np.array([0.5,1.0,1.5,2.0,2.5])
    setup_list  = [setup1,setup2,setup3,setup4,setup5,setup6]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,4]) 
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
mnormas_disp1        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec1         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select1 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
timev_disp1          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec1           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv1             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup2
mnormas_disp2        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec2         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select2 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
timev_disp2          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec2           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv2             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup3
mnormas_disp3        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec3         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select3 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
timev_disp3          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec3           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv3             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup4
mnormas_disp4        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec4         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select4 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
timev_disp4          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec4           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv4             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup5
mnormas_disp5        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec5         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select5 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
timev_disp5          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec5           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv5             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

setup                = setup6
mnormas_disp6        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec6         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select6 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
timev_disp6          = np.load('%stimev_disp_%d.npy'%(setup[0],setup[3]))
timev_rec6           = np.load('%stimev_rec_%d.npy'%(setup[0],setup[3]))
ordersv6             = np.load('%sordersv_%d.npy'%(setup[0],setup[3]))

mnormas_disp        = np.zeros((6,mnormas_disp1.shape[0],mnormas_disp1.shape[1]))
mnormas_disp[0,:,:] = mnormas_disp1[:,:]
mnormas_disp[1,:,:] = mnormas_disp2[:,:]
mnormas_disp[2,:,:] = mnormas_disp3[:,:]
mnormas_disp[3,:,:] = mnormas_disp4[:,:]
mnormas_disp[4,:,:] = mnormas_disp5[:,:]
mnormas_disp[5,:,:] = mnormas_disp6[:,:]

mnormas_rec                              = np.zeros((6,mnormas_rec1.shape[0],mnormas_rec1.shape[1]))
mnormas_rec[0,:,0:mnormas_rec1.shape[1]] = mnormas_rec1[:,:]
mnormas_rec[1,:,0:mnormas_rec2.shape[1]] = mnormas_rec2[:,:]
mnormas_rec[2,:,0:mnormas_rec3.shape[1]] = mnormas_rec3[:,:]
mnormas_rec[3,:,0:mnormas_rec4.shape[1]] = mnormas_rec4[:,:]
mnormas_rec[4,:,0:mnormas_rec5.shape[1]] = mnormas_rec5[:,:]
mnormas_rec[5,:,0:mnormas_rec6.shape[1]] = mnormas_rec6[:,:]

mnormas_disp_select        = np.zeros((6,mnormas_disp_select1.shape[0],mnormas_disp_select1.shape[1]))
mnormas_disp_select[0,:,:] = mnormas_disp_select1[:,:]
mnormas_disp_select[1,:,:] = mnormas_disp_select2[:,:]
mnormas_disp_select[2,:,:] = mnormas_disp_select3[:,:]
mnormas_disp_select[3,:,:] = mnormas_disp_select4[:,:]
mnormas_disp_select[4,:,:] = mnormas_disp_select5[:,:]
mnormas_disp_select[5,:,:] = mnormas_disp_select6[:,:]

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

    for j in range(0,mnormas_disp[i].shape[1]):

        if(timevalue==timev_disp[i][j]): timepos[i] = j
    
    for j in range(0,mnormas_rec[i].shape[1]):

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
# Plotando Resultados - Rotina 1
#==============================================================================
def plot1(mnormas_disp,timev_disp,ordersv,list_scheme,vnames,vdts_select,times_cho,timepos,normtype,ptype,figsave):

    time_disp = (10**-3)*timev_disp

    nscheme = len(vnames)

    plot.figure(figsize = (12,12))
    
    if(normtype==2): plot.suptitle('Quadratic Error of Full Displacement at Time = %.2f s by dt'%time_disp[0][int(timepos[0])]) 
    if(normtype==np.inf): plot.suptitle('Maximum Error Full Displacement at Time = %.2f s by dt'%time_disp[0][int(timepos[0])])
          
    grid = plot.GridSpec(2,2,wspace=0.3,hspace=0.5)
 
    position_plot_listx = np.array([0,0,1,1])
    position_plot_listy = np.array([0,1,0,1])
    ntimes              = len(vdts_select)

    limit_value  = 1 
    vticks       = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline        = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors      = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']

    for k in range(0,ntimes):
        
        max_min_list = []
        kposition    = int(times_cho[k])
        xpos         = int(position_plot_listx[k])
        ypos         = int(position_plot_listy[k])
        nvdt         = kposition
        
        plot.subplot(grid[xpos,ypos])
    
        for i in range(0,nscheme):
    
            listm      = list_scheme[i]
        
            ntestesloc = len(listm)
        
            list_norms = []
        
            for j in range(0,ntestesloc):
            
                index      = int(listm[j][-1])
                posfinal   = int(timepos[kposition])
                norm_value = mnormas_disp[nvdt,index,posfinal]
                                
                if(norm_value<limit_value):
                
                    list_norms.append(norm_value)
                    max_min_list.append(norm_value)
                
                else:
                    
                    list_norms.append(np.nan)
                        
            plot.plot(ordersv[nvdt],list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
        
        max_value = 0
        min_value = 0

        if(len(max_min_list)!=0):
        
            max_value = 1.2*np.amax(max_min_list)
            min_value = 0.8*np.amin(max_min_list)
        
        plot.grid()
        plot.title('dt = %.3f ms'%vdts[nvdt])
        if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
        plot.xticks(ordersv[nvdt])
        plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot.ylim((min_value,max_value))
        ax = plot.gca()
        ax.set_yscale('symlog')

        if(xpos==0 and ypos==0):

            ax.axes.xaxis.set_ticklabels([])
            ax.set_yticks([],minor=True)
            plot.ylabel('Error')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

        if(xpos==0 and ypos==1):

            ax.set_yticks([],minor=True)
            #ax.yaxis.set_ticklabels([])
            #plot.minorticks_off()
            ax.axes.xaxis.set_ticklabels([])
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

        if(xpos==1 and ypos==0):
            
            plot.xlabel('Order')
            plot.ylabel('Error')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
            ax.set_yticks([],minor=True)

        if(xpos==1 and ypos==1):

            ax.set_yticks([],minor=True)
            #ax.yaxis.set_ticklabels([])
            #plot.minorticks_off()
            plot.xlabel('Order')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

        plot.show()
     
    if(normtype==2): plot.savefig('%scomp_methods/norms/plot_norm2_disp_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    if(normtype==np.inf): plot.savefig('%scomp_methods/norms/plot_normmax_disp_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 2
#==============================================================================
def plot2(mnormas_rec,timev_rec,ordersv,list_scheme,vnames,vdts_select,times_cho,setup,timeposrec,normtype,ptype,figsave):

    timev_rec = (10**-3)*timev_rec

    nscheme = len(vnames)

    plot.figure(figsize = (12,12))
    
    if(normtype==2): plot.suptitle('Quadratic Error of Receivers at Time = %.2f s by dt'%timev_rec[0][int(timeposrec[0])]) 
    if(normtype==np.inf): plot.suptitle('Maximum Error Receivers at Time = %.2f s by dt'%timev_rec[0][int(timeposrec[0])])
          
    grid = plot.GridSpec(2,2,wspace=0.3,hspace=0.5)
 
    position_plot_listx = np.array([0,0,1,1])
    position_plot_listy = np.array([0,1,0,1])
    ntimes              = len(vdts_select)
    
    limit_value  = 1    
    vticks       = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline        = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors      = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']

    for k in range(0,ntimes):

        kposition    = int(times_cho[k])        
        xpos         = int(position_plot_listx[k])
        ypos         = int(position_plot_listy[k])
        nvdt         = kposition
        setup        = setup_list[kposition]
        max_min_list = []

        plot.subplot(grid[xpos,ypos])
    
        for i in range(0,nscheme):
    
            listm      = list_scheme[i]
        
            ntestesloc = len(listm)
        
            list_norms = []
        
            for j in range(0,ntestesloc):
            
                index      = listm[j][-1]
                posfinal   = int(timeposrec[kposition])
                norm_value = mnormas_rec[nvdt,index,posfinal]
                
                if(norm_value<limit_value):
                
                    list_norms.append(norm_value)
                    max_min_list.append(norm_value)

                else:
                    
                    list_norms.append(np.nan)         
            
            plot.plot(ordersv[nvdt],list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
        
        max_value = 0
        min_value = 0
        
        if(len(max_min_list)!=0):
  
            max_value = 1.2*np.amax(max_min_list)
            min_value = 0.8*np.amin(max_min_list)
            
        plot.grid()
        plot.title('dt = %.3f ms'%vdts[nvdt])
        if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
        plot.xticks(ordersv[nvdt])
        plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot.ylim((min_value,max_value))
        ax = plot.gca()
        ax.set_yscale('symlog')
        
        if(xpos==0 and ypos==0):
            
            ax.axes.xaxis.set_ticklabels([])
            ax.set_yticks([],minor=True)
            plot.ylabel('Error')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

        if(xpos==0 and ypos==1):

            ax.set_yticks([],minor=True)
            #ax.yaxis.set_ticklabels([])
            #plot.minorticks_off()
            ax.axes.xaxis.set_ticklabels([])
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

        if(xpos==1 and ypos==0):
    
            plot.xlabel('Order')
            plot.ylabel('Error')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
            ax.set_yticks([],minor=True)

        if(xpos==1 and ypos==1):

            ax.set_yticks([],minor=True)
            #ax.yaxis.set_ticklabels([])
            #plot.minorticks_off()
            plot.xlabel('Order')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
        
        plot.show()

    if(normtype==2): plot.savefig('%scomp_methods/norms/plot_norm2_rec_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    if(normtype==np.inf): plot.savefig('%scomp_methods/norms/plot_normmax_rec_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 3
#==============================================================================
def plot3(mnormas_disp_select,timev_disp,ordersv,list_scheme,vnames,vdts_select,times_cho,timepos,xpositionv,ypositionv,normtype,ptype,figsave):

    tn = (10**-3)*timev_disp[0][int(timepos[0])]

    nscheme   = len(vnames)
    nposition = xpositionv.shape[0]

    limit_value  = 1    
    vticks       = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline        = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors      = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']

    for m in range(0,nposition):
        
        max_min_list = []

        plot.figure(figsize = (12,12))
    
        if(normtype==2): plot.suptitle('Quadratic Error of Selected Displacement by dt \n Total Time = %.2f s - Position: x = %.2f m and y = %.2f m'%(tn,xpositionv[m],ypositionv[m]))
        if(normtype==np.inf): plot.suptitle('Maximum Error of Selected Displacement by dt \n Total Time = %.2f s - Position: x = %.2f m and y = %.2f m'%(tn,xpositionv[m],ypositionv[m]))
           
        grid = plot.GridSpec(2,2,wspace=0.1,hspace=0.5)
 
        position_plot_listx = np.array([0,0,1,1])
        position_plot_listy = np.array([0,1,0,1])
        ntimes = len(vdts_select)
    
        for k in range(0,ntimes):
            
            kpostion = int(times_cho[k])
            xpos     = int(position_plot_listx[k])
            ypos     = int(position_plot_listy[k])
            nvdt     = kpostion
        
            plot.subplot(grid[xpos,ypos])
    
            for i in range(0,nscheme):
    
                listm      = list_scheme[i]
                
                ntestesloc = len(listm)
        
                list_norms = []
        
                for j in range(0,ntestesloc):
            
                    index      = listm[j][-1]
                    norm_value = mnormas_disp_select[nvdt,index,m]
                    
                    if(norm_value<limit_value):
                
                        list_norms.append(norm_value)
                        max_min_list.append(norm_value)

                    else:
                    
                        list_norms.append(np.nan)                    
               
                plot.plot(ordersv[nvdt],list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
            
            max_value = 0
            min_value = 0
            
            if(len(max_min_list)!=0):
  
                max_value = 1.2*np.amax(max_min_list)
                min_value = 0.8*np.amin(max_min_list)

            plot.grid()
            plot.title('dt = %.3f ms'%vdts[nvdt])
            if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
            plot.xticks(ordersv[nvdt])
            plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plot.ylim((min_value,max_value))
            ax = plot.gca()
            ax.set_yscale('symlog')           
      
            if(xpos==0 and ypos==0):

                ax.axes.xaxis.set_ticklabels([])
                plot.ylabel('Error')
                ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

            if(xpos==0 and ypos==1):

                ax.set_yticks([],minor=True)
                ax.yaxis.set_ticklabels([])
                plot.minorticks_off()
                ax.axes.xaxis.set_ticklabels([])
                ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

            if(xpos==1 and ypos==0):
                
                plot.xlabel('Order')
                plot.ylabel('Error')
                ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

            if(xpos==1 and ypos==1):

                ax.set_yticks([],minor=True)
                ax.yaxis.set_ticklabels([])
                plot.minorticks_off()
                plot.xlabel('Order')
                ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
            
            plot.show()
     
        if(normtype==2): plot.savefig('%scomp_methods/norms/plot_norm2_x=%.2f_y=%.2f_%d.png'%(figsave,xpositionv[m],ypositionv[m],ptype),dpi=200,bbox_inches='tight')
        if(normtype==np.inf): plot.savefig('%scomp_method/norms/plot_normmax_x=%.2f_y=%.2f_%d.png'%(figsave,xpositionv[m],ypositionv[m],ptype),dpi=200,bbox_inches='tight')
        plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 4
#==============================================================================
def plot4(mnormas_disp,timev_disp,ordersv,list_scheme,vnames,vdts,orders_cho,timepos,normtype,ptype,figsave):

    time_disp = (10**-3)*timev_disp
    nscheme   = len(vnames)
    
    limit_value  = 1    
    vticks       = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline        = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors      = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']
    
    plot.figure(figsize = (12,12))
    
    if(normtype==2): plot.suptitle('Quadratic Error of Full Displacement at Time = %.2f s by Order'%time_disp[0][int(timepos[0])]) 
    if(normtype==np.inf): plot.suptitle('Maximum Error of Full Displacement at Time = %.2f s by Order'%time_disp[0][int(timepos[0])])
          
    grid = plot.GridSpec(2,2,wspace=0.3,hspace=0.5)
 
    position_plot_listx = np.array([0,0,1,1])
    position_plot_listy = np.array([0,1,0,1])
    norders = len(orders_cho)
    ntimes  = len(vdts) 
    
    for k in range(0,norders):
        
        xpos = int(position_plot_listx[k])
        ypos = int(position_plot_listy[k])
        index_order = orders_cho[k]
        max_min_list = []

        plot.subplot(grid[xpos,ypos])
    
        for i in range(0,nscheme):
    
            listm      = list_scheme[i]
                
            list_norms = []
        
            for j in range(0,ntimes):
            
                index      = listm[index_order][-1]
                posfinal   = int(timepos[j])
                norm_value = mnormas_disp[j,index,posfinal]
               
                if(norm_value<limit_value):
                
                    list_norms.append(norm_value)
                    max_min_list.append(norm_value)

                else:
                    
                    list_norms.append(np.nan)    
      
            plot.plot(vdts,list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
        
        theory1 = [t**1 for t in vdts[::-1]]
        correc1 = 1.2*np.amax(max_min_list)
        theory1 = [correc1*th/theory1[0] for th in theory1]        
        plot.plot(vdts[::-1][len(vdts)-2:len(vdts)],theory1[0:2],color='gray')
        plot.annotate('O(dt)', xy=(vdts[::-1][len(vdts)-1], theory1[1]), xytext=(4, 2),textcoords='offset points', size=10,horizontalalignment='left', verticalalignment='top')

        theory2 = [t**2 for t in vdts[::-1]]
        correc2 = 1.2*np.amax(max_min_list)
        theory2 = [correc2*th/theory2[0] for th in theory2]        
        plot.plot(vdts[::-1][len(vdts)-2:len(vdts)],theory2[0:2],color='gray')
        plot.annotate('O(dt^2)', xy=(vdts[::-1][len(vdts)-1], theory2[1]), xytext=(4, 2),textcoords='offset points', size=10,horizontalalignment='left', verticalalignment='top')

        max_value = 0
        min_value = 0

        if(len(max_min_list)!=0):
  
            max_value = 1.2*np.amax(max_min_list)
            min_value = 0.8*np.amin(max_min_list)

        plot.grid()
        ordem = 2*(orders_cho[k]+1)
        plot.title('Order = %d'%(ordem))
        if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
        plot.xticks(vdts)
        plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot.ylim((min_value,max_value))
        ax = plot.gca()
        #ax.set_yscale('symlog')

        if(xpos==0 and ypos==0):

            ax.axes.xaxis.set_ticklabels([])
            ax.set_yticks([],minor=True)
            plot.ylabel('Error')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
            
        if(xpos==0 and ypos==1):

            ax.set_yticks([],minor=True)
           #ax.yaxis.set_ticklabels([])
           #plot.minorticks_off()
            ax.axes.xaxis.set_ticklabels([])
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))


        if(xpos==1 and ypos==0):
            
            plot.xlabel('dt')
            plot.ylabel('Error')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
            ax.set_yticks([],minor=True)

        if(xpos==1 and ypos==1):

            ax.set_yticks([],minor=True)
            #ax.yaxis.set_ticklabels([])
            #plot.minorticks_off()
            plot.xlabel('dt')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
        
        plot.show()
     
    if(normtype==2): plot.savefig('%scomp_methods/norms/plot_norm2_disp_bydt_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    if(normtype==np.inf): plot.savefig('%scomp_methods/norms/plot_normmax_disp_bydt_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 5
#==============================================================================
def plot5(mnormas_rec,timev_rec,ordersv,list_scheme,vnames,vdts,orders_cho,setup,timeposrec,normtype,ptype,figsave):

    time_disp = (10**-3)*timev_rec
    nscheme = len(vnames)
    
    limit_value  = 1    
    vticks       = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline        = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors      = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']
    
    plot.figure(figsize = (12,12))
    
    if(normtype==2): plot.suptitle('Quadratic Error of Full Receivers at Time = %.2f s by Order'%time_disp[0][int(timeposrec[0])]) 
    if(normtype==np.inf): plot.suptitle('Maximum Error of Full Receivers at Time = %.2f s by Order'%time_disp[0][int(timeposrec[0])])
          
    grid = plot.GridSpec(2,2,wspace=0.3,hspace=0.5)
 
    position_plot_listx = np.array([0,0,1,1])
    position_plot_listy = np.array([0,1,0,1])
    norders = len(orders_cho)
    ntimes  = len(vdts) 
    
    for k in range(0,norders):
        
        xpos = int(position_plot_listx[k])
        ypos = int(position_plot_listy[k])
        index_order = orders_cho[k]
        max_min_list = []    

        plot.subplot(grid[xpos,ypos])
    
        for i in range(0,nscheme):
    
            listm      = list_scheme[i]
        
            ntestesloc = len(listm)
        
            list_norms = []
        
            for j in range(0,ntimes):
                
                setup      = setup_list[j]
                index      = listm[index_order][-1]
                posfinal   = int(timeposrec[j])
                norm_value = mnormas_rec[j,index,posfinal]
                
                if(norm_value<limit_value):

                    list_norms.append(norm_value)
                    max_min_list.append(norm_value)

                else:
                    
                    list_norms.append(np.nan)
            
            plot.plot(vdts,list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
        
        theory1 = [t**1 for t in vdts[::-1]]
        correc1 = 1.2*np.amax(max_min_list)
        theory1 = [correc1*th/theory1[0] for th in theory1]        
        plot.plot(vdts[::-1][len(vdts)-2:len(vdts)],theory1[0:2],color='gray')
        plot.annotate('O(dt)', xy=(vdts[::-1][len(vdts)-1], theory1[1]), xytext=(4, 2),textcoords='offset points', size=10,horizontalalignment='left', verticalalignment='top')

        theory2 = [t**2 for t in vdts[::-1]]
        correc2 = 1.2*np.amax(max_min_list)
        theory2 = [correc2*th/theory2[0] for th in theory2]
        plot.plot(vdts[::-1][len(vdts)-2:len(vdts)],theory2[0:2],color='gray')
        plot.annotate('O(dt^2)', xy=(vdts[::-1][len(vdts)-1], theory2[1]), xytext=(4, 2),textcoords='offset points', size=10,horizontalalignment='left', verticalalignment='top')
        
        max_value = 0
        min_value = 0

        if(len(max_min_list)!=0):
  
            max_value = 1.2*np.amax(max_min_list)
            min_value = 0.8*np.amin(max_min_list)

        plot.grid()
        ordem = 2*(orders_cho[k]+1)
        plot.title('Order = %d'%(ordem))
        if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
        plot.xticks(vdts)
        plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot.ylim((min_value,max_value))
        ax = plot.gca()
        #ax.set_yscale('symlog')

        if(xpos==0 and ypos==0):

            ax.axes.xaxis.set_ticklabels([])
            ax.set_yticks([],minor=True)
            plot.ylabel('Error')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

        if(xpos==0 and ypos==1):

            ax.set_yticks([],minor=True)
            #ax.yaxis.set_ticklabels([])
            #plot.minorticks_off()
            ax.axes.xaxis.set_ticklabels([])
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

        if(xpos==1 and ypos==0):
            
            plot.xlabel('dt')
            plot.ylabel('Error')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
            ax.set_yticks([],minor=True)

        if(xpos==1 and ypos==1):

            ax.set_yticks([],minor=True)
            #ax.yaxis.set_ticklabels([])
            #plot.minorticks_off()
            plot.xlabel('dt')
            ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))
        
        plot.show()
     
    if(normtype==2): plot.savefig('%scomp_methods/norms/plot_norm2_rec_bydt_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    if(normtype==np.inf): plot.savefig('%scomp_methods/norms/plot_normmax_rec_bydt_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 6
#==============================================================================
def plot6(mnormas_disp_select,timev_disp,ordersv,list_scheme,vnames,vdts,orders_cho,timepos,xpositionv,ypositionv,normtype,ptype,figsave):

    tn = (10**-3)*timev_disp[0][int(timepos[0])]
    
    limit_value  = 1    
    vticks       = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline        = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors      = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']
    
    nscheme   = len(vnames)
    nposition = xpositionv.shape[0]

    for m in range(0,nposition):
        
        max_min_list = []

        plot.figure(figsize = (12,12))
    
        if(normtype==np.inf): plot.suptitle('Quadratic Error of Selected Displacement by Order \n Total Time = %.2f s Position: x = %.2f m and y = %.2f m'%(tn,xpositionv[m],ypositionv[m]))
        if(normtype==2): plot.suptitle('Quadratic Error of Selected Displacement by Order \n Total Time = %.2f s Position: x = %.2f m and y = %.2f m'%(tn,xpositionv[m],ypositionv[m]))
           
        grid = plot.GridSpec(2,2,wspace=0.1,hspace=0.5)
 
        position_plot_listx = np.array([0,0,1,1])
        position_plot_listy = np.array([0,1,0,1])
        ntimes  = len(vdts)
        norders = len(orders_cho)
    
        for k in range(0,norders):
        
            xpos = int(position_plot_listx[k])
            ypos = int(position_plot_listy[k])
            index_order = orders_cho[k]   
            
            plot.subplot(grid[xpos,ypos])
    
            for i in range(0,nscheme):
    
                listm      = list_scheme[i]
                
                ntestesloc = len(listm)
        
                list_norms = []
        
                for j in range(0,ntimes):
            
                    index      = listm[index_order][-1]
                    norm_value = mnormas_disp_select[j,index,m]
                
                    if(norm_value<limit_value):
                
                        list_norms.append(norm_value)
                        max_min_list.append(norm_value)

                    else:
                    
                        list_norms.append(np.nan)                    
                
                plot.plot(vdts,list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
            
            max_value = 0
            min_value = 0

            if(len(max_min_list)!=0):
  
                max_value = 1.2*np.amax(max_min_list)
                min_value = 0.8*np.amin(max_min_list)
            
            plot.grid()
            ordem = 2*(orders_cho[k]+1)
            plot.title('Order = %d'%(ordem))
            if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
            plot.xticks(vdts)
            plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plot.ylim((min_value,max_value))
            ax = plot.gca()
            ax.set_yscale('symlog')
  
            if(xpos==0 and ypos==0):

                ax.axes.xaxis.set_ticklabels([])
                plot.ylabel('Error')
                ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

            if(xpos==0 and ypos==1):
                
                ax.xaxis.set_ticklabels([])
                ax.set_yticks([],minor=True)
                ax.yaxis.set_ticklabels([])
                plot.minorticks_off()
                ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

            if(xpos==1 and ypos==0):
                
                plot.xlabel('dt [ms]')
                plot.ylabel('Error')
                ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

            if(xpos==1 and ypos==1):

                ax.set_yticks([],minor=True)
                ax.yaxis.set_ticklabels([])
                plot.minorticks_off()
                plot.xlabel('dt [ms]')
                ax.set_yticks(np.round(np.linspace(min_value, max_value, 5), 2))

            plot.show()
     
        if(normtype==2): plot.savefig('%scomp_methods/norms/plot_norm2_bydt_x=%.2f_y=%.2f_%d.png'%(figsave,xpositionv[m],ypositionv[m],ptype),dpi=200,bbox_inches='tight')
        if(normtype==np.inf): plot.savefig('%scomp_method/norms/plot_normmax_bydt_x=%.2f_y=%.2f_%d.png'%(figsave,xpositionv[m],ypositionv[m],ptype),dpi=200,bbox_inches='tight')
        plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados 
#==============================================================================
P1 = plot1(mnormas_disp,timev_disp,ordersv,list_scheme,vnames,vdts_select,times_cho,timepos,normtype,ptype,figsave)
P2 = plot2(mnormas_rec,timev_rec,ordersv,list_scheme,vnames,vdts_select,times_cho,setup,timeposrec,normtype,ptype,figsave)
P3 = plot3(mnormas_disp_select,timev_disp,ordersv,list_scheme,vnames,vdts_select,times_cho,timepos,xpositionv,ypositionv,normtype,ptype,figsave)
P4 = plot4(mnormas_disp,timev_disp,ordersv,list_scheme,vnames,vdts,orders_cho,timepos,normtype,ptype,figsave)
P5 = plot5(mnormas_rec,timev_rec,ordersv,list_scheme,vnames,vdts,orders_cho,setup,timeposrec,normtype,ptype,figsave)
P6 = plot6(mnormas_disp_select,timev_disp,ordersv,list_scheme,vnames,vdts,orders_cho,timepos,xpositionv,ypositionv,normtype,ptype,figsave)
#==============================================================================
