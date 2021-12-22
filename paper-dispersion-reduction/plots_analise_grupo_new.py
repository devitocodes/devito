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
ptype    = 1
normtype = 2

if(ptype==1): 
        
    nptx        = 101 
    npty        = 101    
    t0          = 0      
    tn          = 2000
    nrefesp     = 10
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])
    timevalue   = 600
    setup1      = ('data_save/teste1/dt1/',4000,5,1)
    setup2      = ('data_save/teste1/dt2/',2000,10,2)
    setup3      = ('data_save/teste1/dt3/',1250,16,3)
    setup4      = ('data_save/teste1/dt4/',1000,20,4)
    setup5      = ('data_save/teste1/dt5/',800,25,5)
    figsave     = 'figures/teste1/'
    vdts        = np.array([0.5,1.0,1.6,2.0,2.5])
    setup_list  = [setup1,setup2,setup3,setup4,setup5]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,3,4]) 
    domain_setup = (0,2000,0,2000,0,2000)

if(ptype==2): 
    
    nptx        = 601 
    npty        = 201    
    t0          = 0      
    tn          = 4000
    nrefesp     = 5
    xpositionv  = np.array([4000.0,4000.0,4000.0,6000.0,6000.0,6000.0,8000.0,8000.0,8000.0,])
    ypositionv  = np.array([2000.0,2500.0,1500.0,3000.0,2000.0,2500.0,1500.0,3000.0,2000.0,2500.0,1500.0,3000.0])    
    timevalue   = 1600
    setup1      = ('data_save/teste2/dt1/',10000,2,1)
    setup2      = ('data_save/teste2/dt2/',4000,5,2)
    setup3      = ('data_save/teste2/dt3/',2500,8,3)
    setup4      = ('data_save/teste2/dt4/',2000,10,4)
    setup5      = ('data_save/teste2/dt5/',1666,12,5)
    figsave     = 'figures/teste2/'
    vdts        = np.array([0.4,1.0,1.6,2.0,2.4])
    setup_list  = [setup1,setup2,setup3,setup4,setup5]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,2,3]) 
    domain_setup = (0,12000,0,4000,0,4000)

if(ptype==3): 
    
    nptx        = 201 
    npty        = 201    
    t0          = 0      
    tn          = 2000
    nrefesp     = 5
    xpositionv  = np.array([500.0,1500.0,500.0,1500.0])
    ypositionv  = np.array([500.0,500.0,1500.0,1500.0])   
    timevalue   = 500
    setup1      = ('data_save/teste3/dt1/',4000,5,1)
    setup2      = ('data_save/teste3/dt2/',2000,10,2)
    setup3      = ('data_save/teste3/dt3/',1250,16,3)
    setup4      = ('data_save/teste3/dt4/',1000,20,4)
    setup5      = ('data_save/teste3/dt5/',800,25,5)
    figsave     = 'figures/teste3/'
    vdts        = np.array([0.5,1.0,1.6,2.0,2.5])
    setup_list  = [setup1,setup2,setup3,setup4,setup5]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,2,3]) 
    domain_setup = (0,2000,0,2000,0,2000)
    
if(ptype==4): 
    
    nptx        = 401 
    npty        = 311    
    t0          = 0      
    tn          = 3000
    nrefesp     = 5
    xpositionv  = np.array([30000.0,30000.0,30000.0,40000.0,40000.0,40000.0])
    ypositionv  = np.array([2500.0,5000.0,7500.0,2500.0,5000.0,7500.0,2500.0,5000.0,7500.0])
    timevalue   = 3000
    setup1      = ('data_save/teste4/dt1/',7500,2,1)
    setup2      = ('data_save/teste4/dt2/',3000,5,2)
    setup3      = ('data_save/teste4/dt3/',1875,8,3)
    setup4      = ('data_save/teste4/dt4/',1500,10,4)
    setup5      = ('data_save/teste4/dt5/',1250,12,5)
    figsave     = 'figures/teste4/'
    vdts        = np.array([0.4,1.0,1.6,2.0,2.4])
    setup_list  = [setup1,setup2,setup3,setup4,setup5]
    orders_cho  = np.array([1,3,5,7]) # 4, 8, 12, 16
    times_cho   = np.array([0,1,2,3]) 
    domain_setup = (25000,45000,0,9920,0,3000)
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
mnormas_disp2        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec2         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select2 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
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
mnormas_disp3        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec3         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select3 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
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
mnormas_disp4        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec4         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select4 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
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
mnormas_disp5        = np.load('%smnormas_disp_%d.npy'%(setup[0],setup[3]))
mnormas_rec5         = np.load('%smnormas_rec_%d.npy'%(setup[0],setup[3]))
mnormas_disp_select5 = np.load('%smnormas_disp_select_%d.npy'%(setup[0],setup[3]))
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

mnormas_disp        = np.zeros((5,mnormas_disp1.shape[0],mnormas_disp1.shape[1]))
mnormas_disp[0,:,:] = mnormas_disp1[:,:]
mnormas_disp[1,:,:] = mnormas_disp2[:,:]
mnormas_disp[2,:,:] = mnormas_disp3[:,:]
mnormas_disp[3,:,:] = mnormas_disp4[:,:]
mnormas_disp[4,:,:] = mnormas_disp5[:,:]

mnormas_rec                              = np.zeros((5,mnormas_rec1.shape[0],mnormas_rec1.shape[1]))
mnormas_rec[0,:,0:mnormas_rec1.shape[1]] = mnormas_rec1[:,:]
mnormas_rec[1,:,0:mnormas_rec2.shape[1]] = mnormas_rec2[:,:]
mnormas_rec[2,:,0:mnormas_rec3.shape[1]] = mnormas_rec3[:,:]
mnormas_rec[3,:,0:mnormas_rec4.shape[1]] = mnormas_rec4[:,:]
mnormas_rec[4,:,0:mnormas_rec5.shape[1]] = mnormas_rec5[:,:]

mnormas_disp_select        = np.zeros((5,mnormas_disp_select1.shape[0],mnormas_disp_select1.shape[1]))
mnormas_disp_select[0,:,:] = mnormas_disp_select1[:,:]
mnormas_disp_select[1,:,:] = mnormas_disp_select2[:,:]
mnormas_disp_select[2,:,:] = mnormas_disp_select3[:,:]
mnormas_disp_select[3,:,:] = mnormas_disp_select4[:,:]
mnormas_disp_select[4,:,:] = mnormas_disp_select5[:,:]

rec_num                            = np.zeros((5,rec_num1.shape[0],rec_num1.shape[1],rec_num1.shape[2]))
rec_num[0,:,0:rec_num1.shape[1],:] = rec_num1[:,:,:]
rec_num[1,:,0:rec_num2.shape[1],:] = rec_num2[:,:,:]
rec_num[2,:,0:rec_num3.shape[1],:] = rec_num3[:,:,:]
rec_num[3,:,0:rec_num4.shape[1],:] = rec_num4[:,:,:]
rec_num[4,:,0:rec_num5.shape[1],:] = rec_num5[:,:,:]

solplot            = np.zeros((5,solplot1.shape[0],solplot1.shape[1],solplot1.shape[2],solplot1.shape[3]))
solplot[0,:,:,:,:] = solplot1[:,:,:,:]
solplot[1,:,:,:,:] = solplot2[:,:,:,:]
solplot[2,:,:,:,:] = solplot3[:,:,:,:]
solplot[3,:,:,:,:] = solplot4[:,:,:,:]
solplot[4,:,:,:,:] = solplot5[:,:,:,:]

rec_select_num                                   = np.zeros((5,rec_select_num1.shape[0],rec_select_num1.shape[1],rec_select_num1.shape[2]))
rec_select_num[0,:,0:rec_select_num1.shape[1],:] = rec_select_num1[:,:,:]
rec_select_num[1,:,0:rec_select_num2.shape[1],:] = rec_select_num2[:,:,:]
rec_select_num[2,:,0:rec_select_num3.shape[1],:] = rec_select_num3[:,:,:]
rec_select_num[3,:,0:rec_select_num4.shape[1],:] = rec_select_num4[:,:,:]
rec_select_num[4,:,0:rec_select_num5.shape[1],:] = rec_select_num5[:,:,:]
 
solplot_select                                   = np.zeros((5,solplot_select1.shape[0],solplot_select1.shape[1],solplot_select1.shape[2]))
solplot_select[0,:,:,0:solplot_select1.shape[2]] = solplot_select1[:,:,:]
solplot_select[1,:,:,0:solplot_select2.shape[2]] = solplot_select2[:,:,:]
solplot_select[2,:,:,0:solplot_select3.shape[2]] = solplot_select3[:,:,:]
solplot_select[3,:,:,0:solplot_select4.shape[2]] = solplot_select4[:,:,:]
solplot_select[4,:,:,0:solplot_select5.shape[2]] = solplot_select5[:,:,:]

solplot_ref          = np.zeros((5,solplot_ref1.shape[0],solplot_ref1.shape[1],solplot_ref1.shape[2]))
solplot_ref[0,:,:,:] = solplot_ref1[:,:,:]
solplot_ref[1,:,:,:] = solplot_ref2[:,:,:]
solplot_ref[2,:,:,:] = solplot_ref3[:,:,:]
solplot_ref[3,:,:,:] = solplot_ref4[:,:,:]
solplot_ref[4,:,:,:] = solplot_ref5[:,:,:]
 
rec_ref                          = np.zeros((5,rec_ref1.shape[0],rec_ref1.shape[1]))
rec_ref[0,0:rec_ref1.shape[0],:] = rec_ref1[:,:]
rec_ref[1,0:rec_ref2.shape[0],:] = rec_ref2[:,:]
rec_ref[2,0:rec_ref3.shape[0],:] = rec_ref3[:,:]
rec_ref[3,0:rec_ref4.shape[0],:] = rec_ref4[:,:]
rec_ref[4,0:rec_ref5.shape[0],:] = rec_ref5[:,:]

rec_select_ref                                 = np.zeros((5,rec_select_ref1.shape[0],rec_select_ref1.shape[1]))
rec_select_ref[0,0:rec_select_ref1.shape[0],:] = rec_select_ref1[:,:]
rec_select_ref[1,0:rec_select_ref2.shape[0],:] = rec_select_ref2[:,:]
rec_select_ref[2,0:rec_select_ref3.shape[0],:] = rec_select_ref3[:,:]
rec_select_ref[3,0:rec_select_ref4.shape[0],:] = rec_select_ref4[:,:]
rec_select_ref[4,0:rec_select_ref5.shape[0],:] = rec_select_ref5[:,:]
 
timev_disp      = np.zeros((5,timev_disp1.shape[0]))
timev_disp[0,:] = timev_disp1[:]
timev_disp[1,:] = timev_disp2[:]
timev_disp[2,:] = timev_disp3[:]
timev_disp[3,:] = timev_disp4[:]
timev_disp[4,:] = timev_disp5[:]

timev_rec                          = np.zeros((5,timev_rec1.shape[0]))
timev_rec[0,0:timev_rec1.shape[0]] = timev_rec1[:]
timev_rec[1,0:timev_rec2.shape[0]] = timev_rec2[:]
timev_rec[2,0:timev_rec3.shape[0]] = timev_rec3[:]
timev_rec[3,0:timev_rec4.shape[0]] = timev_rec4[:]
timev_rec[4,0:timev_rec5.shape[0]] = timev_rec5[:]

ordersv      = np.zeros((5,ordersv1.shape[0]))
ordersv[0,:] = ordersv1[:]
ordersv[1,:] = ordersv2[:]
ordersv[2,:] = ordersv3[:]
ordersv[3,:] = ordersv4[:]
ordersv[4,:] = ordersv5[:]

vnames = ['Classic', 'Cross2009', 'Cross2013', 'Cross2016_TE', 'Cross2016_LS', 'Cross_Rb2016_TE', 
          'Cross_Rb2016_LS', 'Rhombus2016_TE', 'RHombus2016_LS']

vdts_select  = np.zeros(times_cho.shape[0])

for k in range(0,times_cho.shape[0]):
        
    vdts_select[k] = vdts[int(times_cho[k])]
#==============================================================================

#==============================================================================
# Manipulando Solução de Referência
#==============================================================================
timepos    = np.zeros(5)
timeposrec = np.zeros(5)

for i in range(0,5):

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
def plot1(mnormas_disp,timev_disp,ordersv,list_scheme,vnames,vdts_select,times_cho,normtype,ptype,figsave):

    time_disp = (10**-3)*timev_disp

    nscheme = len(vnames)

    plot.figure(figsize = (12,12))
    
    if(normtype==2): plot.suptitle('Quadratic Error of Full Displacement at Final Time = %.2f s by dt'%time_disp[0][-1]) 
    if(normtype==np.inf): plot.suptitle('Maximum Error Full Displacement at Final Time = %.2f s by dt'%time_disp[0][-1])
          
    grid = plot.GridSpec(2,2,wspace=0.1,hspace=0.5)
 
    position_plot_listx = np.array([0,0,1,1])
    position_plot_listy = np.array([0,1,0,1])
    
    ntimes = len(vdts_select)
    
    min_value =  100000000
    max_value = -100000000
    
    for m1 in range(0,mnormas_disp.shape[0]):
        
        for m2 in range(0,mnormas_disp.shape[1]):
    
            value = mnormas_disp[m1,m2,-1]
            
            if((np.isfinite(value)==True) and (value>0) and (value<min_value)): min_value = value
            
            if((np.isfinite(value)==True) and (value<1) and (value>max_value)): max_value = value
    
    min_value =  0.8*min_value
    max_value =  1.4*max_value
    
    vticks    = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline     = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors   = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']

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
                norm_value = mnormas_disp[nvdt,index,-1]

                if(norm_value<1):
                
                    list_norms.append(norm_value)
                
                else:
                    
                    list_norms.append(np.nan)
                    
            plot.plot(ordersv[nvdt],list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])

        plot.grid()
        plot.title('dt = %.3f ms'%vdts[nvdt])
        if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
        plot.xticks(ordersv[nvdt])
        plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot.ylim((min_value,max_value))
        ax = plot.gca()
        ax.set_yscale('log')

        if(xpos==0 and ypos==0):

            ax.axes.xaxis.set_ticklabels([])
            plot.ylabel('Error')
            
        if(xpos==0 and ypos==1):

            ax.set_yticks([],minor=True)
            ax.yaxis.set_ticklabels([])
            plot.minorticks_off()
            ax.axes.xaxis.set_ticklabels([])

        if(xpos==1 and ypos==0):
            
            plot.xlabel('Order')
            plot.ylabel('Error')

        if(xpos==1 and ypos==1):

            ax.set_yticks([],minor=True)
            ax.yaxis.set_ticklabels([])
            plot.minorticks_off()
            plot.xlabel('Order')

        plot.show()
     
    if(normtype==2): plot.savefig('%scomp_methods/plot_norm2_disp_tf_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    if(normtype==np.inf): plot.savefig('%scomp_methods/plot_normmax_disp_tf_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 2
#==============================================================================
def plot2(mnormas_rec,timev_rec,ordersv,list_scheme,vnames,vdts_select,times_cho,setup,normtype,ptype,figsave):

    timev_rec = (10**-3)*timev_rec

    nscheme = len(vnames)

    plot.figure(figsize = (12,12))
    
    if(normtype==2): plot.suptitle('Quadratic Error of Receivers at Final Time = %.2f s by dt'%timev_rec[0][-1]) 
    if(normtype==np.inf): plot.suptitle('Maximum Error Receivers at Final Time = %.2f s by dt'%timev_rec[0][-1])
          
    grid = plot.GridSpec(2,2,wspace=0.1,hspace=0.5)
 
    position_plot_listx = np.array([0,0,1,1])
    position_plot_listy = np.array([0,1,0,1])
    ntimes = len(vdts_select)
    
    min_value =  100000000
    max_value = -100000000
    
    for m1 in range(0,mnormas_rec.shape[0]):
        
        for m2 in range(0,mnormas_rec.shape[1]):
    
            value = mnormas_rec[m1,m2,-1]
            
            if((np.isfinite(value)==True) and (value>0) and (value<min_value)): min_value = value
            
            if((np.isfinite(value)==True) and (value<1) and (value>max_value)): max_value = value
    
    min_value =  0.8*min_value
    max_value =  1.4*max_value
    vticks    = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline     = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors   = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']

    for k in range(0,ntimes):

        kpostion = int(times_cho[k])        
        xpos     = int(position_plot_listx[k])
        ypos     = int(position_plot_listy[k])
        nvdt     = kpostion
        setup    = setup_list[kpostion]
        
        plot.subplot(grid[xpos,ypos])
    
        for i in range(0,nscheme):
    
            listm      = list_scheme[i]
        
            ntestesloc = len(listm)
        
            list_norms = []
        
            for j in range(0,ntestesloc):
            
                index      = listm[j][-1]
                posfinal   = setup[1]
                norm_value = mnormas_rec[nvdt,index,posfinal]
                
                if(norm_value<1):
                
                    list_norms.append(norm_value)
                
                else:
                    
                    list_norms.append(np.nan)         
                    
            plot.plot(ordersv[nvdt],list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
    
        plot.grid()
        plot.title('dt = %.3f ms'%vdts[nvdt])
        if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
        plot.xticks(ordersv[nvdt])
        plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot.ylim((min_value,max_value))
        ax = plot.gca()
        ax.set_yscale('log')
        
        if(xpos==0 and ypos==0):

            ax.axes.xaxis.set_ticklabels([])
            plot.ylabel('Error')

        if(xpos==0 and ypos==1):

            ax.set_yticks([],minor=True)
            ax.yaxis.set_ticklabels([])
            plot.minorticks_off()
            ax.axes.xaxis.set_ticklabels([])

        if(xpos==1 and ypos==0):
            
            plot.xlabel('Order')
            plot.ylabel('Error')

        if(xpos==1 and ypos==1):

            ax.set_yticks([],minor=True)
            ax.yaxis.set_ticklabels([])
            plot.minorticks_off()
            plot.xlabel('Order')
        
        plot.show()

    if(normtype==2): plot.savefig('%scomp_methods/plot_norm2_rec_tf_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    if(normtype==np.inf): plot.savefig('%scomp_methods/plot_normmax_rec_tf_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 3
#==============================================================================
def plot3(mnormas_disp_select,timev_disp,ordersv,list_scheme,vnames,vdts_select,times_cho,xpositionv,ypositionv,normtype,ptype,figsave):

    tn = (10**-3)*timev_disp[0][-1]

    nscheme   = len(vnames)
    nposition = xpositionv.shape[0]

    min_value =  100000000
    max_value = -100000000
    
    for m1 in range(0,mnormas_disp_select.shape[0]):

        for m2 in range(0,mnormas_disp_select.shape[1]):
        
            for m3 in range(0,mnormas_disp_select.shape[2]):
    
                value = mnormas_disp_select[m1,m2,m3]
            
                if((np.isfinite(value)==True) and (value>0) and (value<min_value)): min_value = value
                
                if((np.isfinite(value)==True) and (value<1) and (value>max_value)): max_value = value
    
    min_value =  0.8*min_value
    max_value =  1.4*max_value
    vticks    = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline     = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors   = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']

    for m in range(0,nposition):

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
                    
                    if(norm_value<1):
                
                        list_norms.append(norm_value)
                
                    else:
                    
                        list_norms.append(np.nan)                    
               
                plot.plot(ordersv[nvdt],list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
            
            plot.grid()
            plot.title('dt = %.3f ms'%vdts[nvdt])
            if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
            plot.xticks(ordersv[nvdt])
            plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plot.ylim((min_value,max_value))
            ax = plot.gca()
            ax.set_yscale('log')           
      
            if(xpos==0 and ypos==0):

                ax.axes.xaxis.set_ticklabels([])
                plot.ylabel('Error')

            if(xpos==0 and ypos==1):

                ax.set_yticks([],minor=True)
                ax.yaxis.set_ticklabels([])
                plot.minorticks_off()
                ax.axes.xaxis.set_ticklabels([])

            if(xpos==1 and ypos==0):
                
                plot.xlabel('Order')
                plot.ylabel('Error')

            if(xpos==1 and ypos==1):

                ax.set_yticks([],minor=True)
                ax.yaxis.set_ticklabels([])
                plot.minorticks_off()
                plot.xlabel('Order')            
            
            plot.show()
     
        if(normtype==2): plot.savefig('%scomp_methods/plot_norm2_x=%.2f_y=%.2f_%d.png'%(figsave,xpositionv[m],ypositionv[m],ptype),dpi=200,bbox_inches='tight')
        if(normtype==np.inf): plot.savefig('%scomp_method/plot_normmax_x=%.2f_y=%.2f_%d.png'%(figsave,xpositionv[m],ypositionv[m],ptype),dpi=200,bbox_inches='tight')
        plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 4
#==============================================================================
def plot4(mnormas_disp,timev_disp,ordersv,list_scheme,vnames,vdts,orders_cho,normtype,ptype,figsave):

    time_disp = (10**-3)*timev_disp
    nscheme = len(vnames)
    
    min_value =  100000000
    max_value = -100000000
    
    for m1 in range(0,mnormas_disp.shape[0]):
        
        for m2 in range(0,mnormas_disp.shape[1]):
    
            value = mnormas_disp[m1,m2,-1]
            
            if((np.isfinite(value)==True) and (value>0) and (value<min_value)): min_value = value
            
            if((np.isfinite(value)==True) and (value<1) and (value>max_value)): max_value = value
   
    min_value =  0.8*min_value
    max_value =  1.4*max_value
    vticks    = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline     = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors   = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']
    
    plot.figure(figsize = (12,12))
    
    if(normtype==2): plot.suptitle('Quadratic Error of Full Displacement at Final Time = %.2f s by Order'%time_disp[0][-1]) 
    if(normtype==np.inf): plot.suptitle('Maximum Error of Full Displacement at Final Time = %.2f s by Order'%time_disp[0][-1])
          
    grid = plot.GridSpec(2,2,wspace=0.1,hspace=0.5)
 
    position_plot_listx = np.array([0,0,1,1])
    position_plot_listy = np.array([0,1,0,1])
    norders = len(orders_cho)
    ntimes  = len(vdts) 
    
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
                norm_value = mnormas_disp[j,index,-1]
               
                if(norm_value<1):
                
                    list_norms.append(norm_value)
                
                else:
                    
                    list_norms.append(np.nan)    
            
            plot.plot(vdts,list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
    
        plot.grid()
        ordem = 2*(orders_cho[k]+1)
        plot.title('Order = %d'%(ordem))
        if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
        plot.xticks(vdts)
        plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot.ylim((min_value,max_value))
        ax = plot.gca()
        ax.set_yscale('log')

        if(xpos==0 and ypos==0):

            ax.axes.xaxis.set_ticklabels([])
            plot.ylabel('Error')

        if(xpos==0 and ypos==1):

            ax.set_yticks([],minor=True)
            ax.yaxis.set_ticklabels([])
            plot.minorticks_off()
            ax.axes.xaxis.set_ticklabels([])

        if(xpos==1 and ypos==0):
            
            plot.xlabel('dt [ms]')
            plot.ylabel('Error')

        if(xpos==1 and ypos==1):

            ax.set_yticks([],minor=True)
            ax.yaxis.set_ticklabels([])
            plot.minorticks_off()
            plot.xlabel('dt [ms]')
        
        plot.show()
     
    if(normtype==2): plot.savefig('%scomp_methods/plot_norm2_disp_tf_bydt_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    if(normtype==np.inf): plot.savefig('%scomp_methods/plot_normmax_disp_tf_bydt_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 5
#==============================================================================
def plot5(mnormas_rec,timev_rec,ordersv,list_scheme,vnames,vdts,orders_cho,setup,normtype,ptype,figsave):

    time_disp = (10**-3)*timev_disp
    nscheme = len(vnames)
    
    min_value =  100000000
    max_value = -100000000
    
    for m1 in range(0,mnormas_rec.shape[0]):
        
        for m2 in range(0,mnormas_rec.shape[1]):
    
            value = mnormas_rec[m1,m2,-1]
            
            if((np.isfinite(value)==True) and (value>0) and (value<min_value)): min_value = value
            
            if((np.isfinite(value)==True) and (value<1) and (value>max_value)): max_value = value
    
    min_value =  0.8*min_value
    max_value =  1.4*max_value
    vticks    = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline     = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors   = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']

    plot.figure(figsize = (12,12))
    
    if(normtype==2): plot.suptitle('Quadratic Error of Full Displacement at Final Time = %.2f s by Order'%time_disp[0][-1]) 
    if(normtype==np.inf): plot.suptitle('Maximum Error of Full Displacement at Final Time = %.2f s by Order'%time_disp[0][-1])
          
    grid = plot.GridSpec(2,2,wspace=0.1,hspace=0.5)
 
    position_plot_listx = np.array([0,0,1,1])
    position_plot_listy = np.array([0,1,0,1])
    norders = len(orders_cho)
    ntimes  = len(vdts) 
    
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
                
                setup      = setup_list[j]
                index      = listm[index_order][-1]
                posfinal   = setup[1]
                norm_value = mnormas_rec[j,index,posfinal]
                
                if(norm_value<1):
                
                    list_norms.append(norm_value)
                
                else:
                    
                    list_norms.append(np.nan)
            
            plot.plot(vdts,list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
    
        plot.grid()
        ordem = 2*(orders_cho[k]+1)
        plot.title('Order = %d'%(ordem))
        if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
        plot.xticks(vdts)
        plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot.ylim((min_value,max_value))
        ax = plot.gca()
        ax.set_yscale('log')

        if(xpos==0 and ypos==0):

            ax.axes.xaxis.set_ticklabels([])
            plot.ylabel('Error')

        if(xpos==0 and ypos==1):

            ax.set_yticks([],minor=True)
            ax.yaxis.set_ticklabels([])
            plot.minorticks_off()
            ax.axes.xaxis.set_ticklabels([])

        if(xpos==1 and ypos==0):
            
            plot.xlabel('dt [ms]')
            plot.ylabel('Error')

        if(xpos==1 and ypos==1):

            ax.set_yticks([],minor=True)
            ax.yaxis.set_ticklabels([])
            plot.minorticks_off()
            plot.xlabel('dt [ms]')
        
        plot.show()
     
    if(normtype==2): plot.savefig('%scomp_methods/plot_norm2_rec_tf_bydt_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    if(normtype==np.inf): plot.savefig('%scomp_methods/plot_normmax_rec_tf_bydt_%d.png'%(figsave,ptype),dpi=200,bbox_inches='tight')
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 6
#==============================================================================
def plot6(mnormas_disp_select,timev_disp,ordersv,list_scheme,vnames,vdts,orders_cho,xpositionv,ypositionv,normtype,ptype,figsave):

    tn = (10**-3)*timev_disp[0][-1]
    
    min_value =  100000000
    max_value = -100000000
    
    for m1 in range(0,mnormas_disp_select.shape[0]):

        for m2 in range(0,mnormas_disp_select.shape[1]):
        
            for m3 in range(0,mnormas_disp_select.shape[2]):
    
                value = mnormas_disp_select[m1,m2,m3]
            
                if((np.isfinite(value)==True) and (value>0) and (value<min_value)): min_value = value
                
                if((np.isfinite(value)==True) and (value<1) and (value>max_value)): max_value = value
    
    min_value =  0.8*min_value
    max_value =  1.4*max_value
    vticks    = ['s', '+', '+', '+',  '+',  '^',   '^',   'D',      'D']
    vline     = ['-', '-', '-', '--', '-.', '--',  '-.',  '--',     '-.']
    vcolors   = ['b', 'g', 'r', 'c',  'm',  'y',   'b',   'purple', 'teal']
    
    nscheme   = len(vnames)
    nposition = xpositionv.shape[0]

    for m in range(0,nposition):

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
                
                    if(norm_value<1):
                
                        list_norms.append(norm_value)
                
                    else:
                    
                        list_norms.append(np.nan)                    
                
                plot.plot(vdts,list_norms,color=vcolors[i],linestyle=vline[i],marker=vticks[i],label=vnames[i])
            
            plot.grid()
            ordem = 2*(orders_cho[k]+1)
            plot.title('Order = %d'%(ordem))
            if(xpos==0 and ypos==0): plot.legend(loc="lower center",ncol=3,bbox_to_anchor=(1.05, -0.4))
            plot.xticks(vdts)
            plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plot.ylim((min_value,max_value))
            ax = plot.gca()
            ax.set_yscale('log')
  
            if(xpos==0 and ypos==0):

                ax.axes.xaxis.set_ticklabels([])
                plot.ylabel('Error')

            if(xpos==0 and ypos==1):
                
                ax.xaxis.set_ticklabels([])
                ax.set_yticks([],minor=True)
                ax.yaxis.set_ticklabels([])
                plot.minorticks_off()

            if(xpos==1 and ypos==0):
                
                plot.xlabel('dt [ms]')
                plot.ylabel('Error')

            if(xpos==1 and ypos==1):

                ax.set_yticks([],minor=True)
                ax.yaxis.set_ticklabels([])
                plot.minorticks_off()
                plot.xlabel('dt [ms]')

            plot.show()
     
        if(normtype==2): plot.savefig('%scomp_methods/plot_norm2_bydt_x=%.2f_y=%.2f_%d.png'%(figsave,xpositionv[m],ypositionv[m],ptype),dpi=200,bbox_inches='tight')
        if(normtype==np.inf): plot.savefig('%scomp_method/plot_normmax_bydt_x=%.2f_y=%.2f_%d.png'%(figsave,xpositionv[m],ypositionv[m],ptype),dpi=200,bbox_inches='tight')
        plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 7
#==============================================================================
def plot7(solplot_ref,solplot,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timepos):
    
    fscale    =  10**(-3)    
    timevalue = fscale*timevalue

    position_plot_listx = np.array([0,0,1,1,2,2,3,3,4,4])
    position_plot_listy = np.array([0,1,0,1,0,1,0,1,0,1])
    ntimes   = vdts.shape[0]
    norders  = ordersv[0].shape[0]
    nschemes = len(list_scheme) + 1
    
    x0    = domain_setup[0]
    x1    = domain_setup[1]
    y0    = domain_setup[2]
    y1    = domain_setup[3]
    scale = max(np.amax(solplot_ref),np.amax(solplot))/50
    
    for k1 in range(0,ntimes):
        
        timeposloc = int(timepos[k1])
    
    #for k1 in range(0,1):

        for k2 in range(0,norders):
        #for k2 in range(0,1):

            fig1 = plot.figure(figsize = (3,8))
   
            plot.suptitle('Displacement - Space Order = %d \n T = %.2f s - dt = %.3f ms'%(ordersv[0][k2],timevalue,vdts[k1]),fontsize=10)
 
            grid = plot.GridSpec(5,2,wspace=0.45,hspace=0.1)
   
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
                fig    = plot.imshow(np.transpose(sol),vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
                plot.grid()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.tick_params(axis="x", labelsize=6)
                ax.tick_params(axis="y", labelsize=6)

                if(ypos==0): 
                   
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    
                if(xpos==4): 
                
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    
                if(xpos==4 and ypos==0): 
                    
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
            
            cb_ax = fig1.add_axes([0.001, 0.06, 1.0 , 0.02])
            cbar = fig1.colorbar(fig, cax=cb_ax,format='%.2e',orientation='horizontal')
            cbar.ax.tick_params(labelsize=6)
            plot.show()
            plot.savefig('%scomp_methods/disp_order_%d_dt_%f_%d.png'%(figsave,ordersv[0][k2],vdts[k1],ptype),dpi=200,bbox_inches='tight')
            plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados - Rotina 8
#==============================================================================
def plot8(rec_ref,rec_num,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timeposrec):
    
    fscale    =  10**(-3)    
    timevalue = fscale*timevalue

    position_plot_listx = np.array([0,0,1,1,2,2,3,3,4,4])
    position_plot_listy = np.array([0,1,0,1,0,1,0,1,0,1])
    ntimes   = vdts.shape[0]
    norders  = ordersv[0].shape[0]
    nschemes = len(list_scheme) + 1
    
    x0    = domain_setup[0]
    x1    = domain_setup[1]
    t0    = domain_setup[4]
    tn    = domain_setup[5]
    scale = max(np.amax(rec_ref),np.amax(rec_num))/50
    
    for k1 in range(0,ntimes):
        
        timeposrecloc = int(timeposrec[k1])
        
    #for k1 in range(0,1):

        for k2 in range(0,norders):
        #for k2 in range(0,1):
               
            fig1 = plot.figure(figsize = (3,8))
   
            plot.suptitle('Receiver - Space Order = %d \n T = %.2f s - dt = %.3f ms'%(ordersv[0][k2],timevalue,vdts[k1]),fontsize=10)
 
            grid = plot.GridSpec(5,2,wspace=0.45,hspace=0.1)
   
            for k3 in range(0,nschemes):
                
                if(k3==0):
                    
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    setup      = setup_list[k1]
                    posfinal   = timeposrecloc#setup[1]
                    rec = rec_ref[k1,0:posfinal,:]
                    plot.title('Reference',fontsize=7)
                    
                else:
                    
                    xpos = int(position_plot_listx[k3])
                    ypos = int(position_plot_listy[k3])
                    plot.subplot(grid[xpos,ypos])           
                    setup      = setup_list[k1]
                    posfinal   = timeposrecloc#setup[1]
                    listm  = list_scheme[k3-1]
                    index  = listm[k2][-1]
                    rec    = rec_num[k1,index,0:posfinal,:]
                    plot.title(vnames[k3-1],fontsize=7)
                
                ax = plot.gca()
                extent = [fscale*x0,fscale*x1, fscale*tn, fscale*t0]
                fig = plot.imshow(rec,vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
                plot.grid()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.tick_params(axis="x", labelsize=6)
                ax.tick_params(axis="y", labelsize=6)

                if(ypos==0): 
                   
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
                    
                if(xpos==4): 
                
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    
                if(xpos==4 and ypos==0): 
            
                    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
                    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
                    
            #fig1.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,wspace=0.02, hspace=0.02)
            cb_ax = fig1.add_axes([0.001, 0.06, 1.0 , 0.02])
            cbar = fig1.colorbar(fig, cax=cb_ax,format='%.2e',orientation='horizontal')
            cbar.ax.tick_params(labelsize=6)
            plot.show()
            plot.savefig('%scomp_methods/rec_order_%d_dt_%f_%d.png'%(figsave,ordersv[0][k2],vdts[k1],ptype),dpi=200,bbox_inches='tight')
            plot.close()

    return
#==============================================================================

#==============================================================================
# Plotando Resultados 
#==============================================================================
P1 = plot1(mnormas_disp,timev_disp,ordersv,list_scheme,vnames,vdts_select,times_cho,normtype,ptype,figsave)
P2 = plot2(mnormas_rec,timev_rec,ordersv,list_scheme,vnames,vdts_select,times_cho,setup,normtype,ptype,figsave)
P3 = plot3(mnormas_disp_select,timev_disp,ordersv,list_scheme,vnames,vdts_select,times_cho,xpositionv,ypositionv,normtype,ptype,figsave)
P4 = plot4(mnormas_disp,timev_disp,ordersv,list_scheme,vnames,vdts,orders_cho,normtype,ptype,figsave)
P5 = plot5(mnormas_rec,timev_rec,ordersv,list_scheme,vnames,vdts,orders_cho,setup,normtype,ptype,figsave)
P6 = plot6(mnormas_disp_select,timev_disp,ordersv,list_scheme,vnames,vdts,orders_cho,xpositionv,ypositionv,normtype,ptype,figsave)
P7 = plot7(solplot_ref,solplot,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timepos)
P8 = plot8(rec_ref,rec_num,domain_setup,ordersv,vdts,list_scheme,vnames,ptype,timevalue,figsave,timeposrec)
#==============================================================================