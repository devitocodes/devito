#==============================================================================
if (__name__=='__main__'):    
#==============================================================================

    #==============================================================================
    # Python Imports
    #==============================================================================
    import numpy                                        as np
    import math                                         as mt
    import time                                         as tm
    import sys
    import segyio
    import matplotlib.pyplot                            as plot
    import psutil
    from   numpy             import linalg              as la
    from   scipy.ndimage     import gaussian_filter
    from   scipy             import optimize
    from   dask.distributed  import Client, wait
    #==============================================================================

    #==============================================================================
    # Devito Imports
    #==============================================================================
    from   devito           import *
    from   examples.seismic import RickerSource
    from   examples.seismic import Receiver
    configuration['log-level']='ERROR'
    #==============================================================================

    #==============================================================================
    # Our Imports
    #==============================================================================
    import settings_config
    sys.path.insert(0, './code')
    from   timeit import default_timer as timer
    import solver, domain2D, utils, velmodel
    #==============================================================================

    #==============================================================================
    plot.close("all")
    #==============================================================================
    
    #==============================================================================
    # Model Properties
    #==============================================================================    
    model=settings_config.settings.model
    #==============================================================================
    
    #==============================================================================
    # Parameters Settings
    #==============================================================================
    if(model['vp']=='Marmousi'):
    
        setting = settings_config.settings.setting2

    elif(model['vp']=='Salt'):

        setting = settings_config.settings.setting4

    elif(model['vp']=='Circle' or model['vp']=='HorizontalLayers'):

        setting = settings_config.settings.setting3
        
    setup  = utils.ProblemSetup(setting)
    #==============================================================================
    
    #==============================================================================
    # Grid Construction
    #==============================================================================
    grid   = domain2D.SetGrid(setup)
    (x, z) = grid.dimensions
    #==============================================================================

    #==============================================================================
    # Chosing the model
    #==============================================================================
    if(model['vp']=='Marmousi'):
    
        with segyio.open('VelModelFiles/Mar2_Vp_1.25m.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        
        v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file,start_model='Initial')

    elif(model['vp']=='Salt'):

        vp_file = np.load("./VelModelFiles/seg_eage_xcut_338.npy")
        v0      = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)
        vo      = 3.0*v0/v0

    elif(model['vp']=='Circle' or model['vp']=='HorizontalLayers'):
        v0      = velmodel.SetVel(model, setup, setting, grid, start_model='Initial')
    #==============================================================================

    #==============================================================================
    # Time Parameters
    #==============================================================================
    set_time = setup.TimeDiscret(v0)
    dt0, nt, time_range = set_time   #time discretization
    #==============================================================================

    #==============================================================================
    # Shot Properties
    #==============================================================================
    sd     = setting["shots_dist"]
    nshots = int((setting["lenx"]-200)/sd)+1
    #==============================================================================

    #==============================================================================
    # Start DASK -  sources paralelization
    #==============================================================================
    if(setting["dask"]): 
        
        from distributed import LocalCluster
        
        cluster = LocalCluster(n_workers=nshots,death_timeout=600)
        client  = Client(cluster)
    #==============================================================================

    #==============================================================================
    # FWI Solver Class
    #==============================================================================
    fwisolver = solver.FWISolver(set_time,setup, setting,grid,utils,v0,test="adjoint")
    objvr     = []
    cont      = 0 
    m0        = np.reshape(v0,-1) 
    
    # Reading the receivers related to the true vel. model
    rec_true = []
    
    for sn in range(0, nshots):  
        
        # Receivers computed in the reference domain  
        filename = "rec_ref.npy"
        rec_true.append(np.load(filename))
    
    def shots(m0):

        global objvr, cont   
        objective = 0.

        work  = []
        grad1 = Function(name="grad1", grid=grid)
        fwisolver.vp_guess(m0)
        vp_guess = fwisolver.vp_g
       
        print("Running adjoint solver.")
        for sn in range(0, nshots):    
            clear_cache()
            fwisolver.rec_true = rec_true[sn]

            if(setting["dask"]):
                
                work.append(client.submit(fwisolver.apply,sn))
            
            else:
                
                adj = fwisolver.apply(sn)

         
        if(setting["dask"]):
            
            wait(work)
            
            for i in range(0, nshots):
            
                adj.append(work[i].result())
                
        return adj

    shots(m0)
    #==============================================================================