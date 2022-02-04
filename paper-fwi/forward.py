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
    from   memory_profiler   import profile
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
    
    setup   = utils.ProblemSetup(setting)
    #==============================================================================

    #==============================================================================
    # Grid Construction
    #==============================================================================
    grid    = domain2D.SetGrid(setup)
    (x, z)  = grid.dimensions
    #==============================================================================

    #==============================================================================
    # Chosing the model
    #==============================================================================
    if(model['vp']=='Marmousi'):
    
        with segyio.open('VelModelFiles/Mar2_Vp_1.25m.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file,start_model='True')
    
    elif(model['vp']=='Salt'):

        vp_file = np.load("./VelModelFiles/seg_eage_xcut_338.npy")
        v0      = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)
        
    elif(model['vp']=='Circle'or model['vp']=='HorizontalLayers'):
        v0      = velmodel.SetVel(model,setup, setting,grid)
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
    fwisolver = solver.FWISolver(set_time,setup, setting,grid,utils,v0)

    def run():
        
        rec_true  = []
        work_true = []
        print("Running forward wave equation")
        
        for sn in range(0, nshots): 
            
            if(setting["dask"]):
        
                work_true.append(client.submit(fwisolver.forward_true,sn))

            else:
                
                rec_data, u_data = fwisolver.forward_true(sn)
                # np.save('u_ref',u_data)
                # np.save('rec_ref',rec_data)

        if(setting["dask"]):
            
            wait(work_true)
            
            for i in range(nshots):
            
                np.save('rec_ref' + str(i),work_true[i].result()[0])
                np.save('u_ref' + str(i),work_true[i].result()[1])
    
    shot_rec = run()
#==============================================================================