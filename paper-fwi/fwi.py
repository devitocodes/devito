#==============================================================================
if (__name__=='__main__'):    
#==============================================================================

    #==============================================================================
    # Python Imports
    #==============================================================================
    import numpy                                        as np
    import sys
    import segyio
    import matplotlib.pyplot                            as plot
    import psutil
    from   scipy.ndimage     import gaussian_filter
    from   scipy             import optimize
    from  dask.distributed   import Client, wait
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
    import solver, domain2D, utils, velmodel
    import plots
    #==============================================================================

    #==============================================================================
    plot.close("all")
    #==============================================================================
    
    #==============================================================================
    # Model Properties
    #==============================================================================    
    model=settings_config.settings.model
    #==============================================================================
    if(model['vp']=='Marmousi'):
    
        setting = settings_config.settings.setting1

    elif(model['vp']=='Salt'):

        setting = settings_config.settings.setting5

    setup   = utils.ProblemSetup(setting)    
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
        v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file,start_model='True')

    elif(model['vp']=='Salt'):

        vp_file = np.load("./VelModelFiles/seg_eage_xcut_338.npy")
        v0      = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)

    sigma = 15
    vini  = gaussian_filter(v0,sigma=sigma)
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
        
        cluster = LocalCluster(n_workers=int(nshots),threads_per_worker=1,death_timeout=600)
        client  = Client(cluster)
   #==============================================================================
   
   #==============================================================================     
   # FWI Solver Class
   #==============================================================================
    fwisolver = solver.FWISolver(set_time,setup, setting,grid,utils,vini)
   #==============================================================================

   #==============================================================================
   # FWI Analisys Variables  
   #==============================================================================
    objvr = []
   #==============================================================================

    #==============================================================================
    # FWI Functions
    #==============================================================================    
    def shots(m0,rec_true):  
        
        objective = 0.

        work  = []
        grad1 = Function(name="grad1", grid=grid)
        fwisolver.vp_guess(m0)
        
        vp_guess = fwisolver.vp_g

        for sn in range(0, nshots):    
            
            clear_cache()
            fwisolver.rec_true = rec_true[sn]
            
            if(setting["dask"]):
                
                work.append(client.submit(fwisolver.apply,sn))
            
            else:
                
                aux0, aux1       = fwisolver.apply(sn)
                objective       += aux0
                grad1.data[:,:] += aux1

        if(setting["dask"]):
            
            wait(work)
            
            for i in range(0, nshots):
            
                objective       += work[i].result()[0]
                grad1.data[:,:] += work[i].result()[1]
                work[i]          = None
        
        grad_grop = np.array(grad1.data[:])[:, :]
        
        if(setting["Abcs"]=='pml'):
            
            vres = vp_guess[0].data 
        
        else:
            
            vres = vp_guess.data 
       
        objvr.append(objective)
    
        np.save('vp_' + str(fwisolver.freq),vres)
        np.savetxt('misfit_'+ str(fwisolver.freq), objvr, delimiter=' ')

        return objective, np.reshape(grad_grop,-1)
   #==============================================================================

   #==============================================================================
   # Multiscale
   #==============================================================================
    ite = setting["fwi_iteration"]

    if(setting['multiscale']):

        freq_bands = setting["freq_bands"]
   
        for freq in freq_bands:

            # Reading true data source
            rec_true = []
            for sn in range(0, nshots):    
                filename = "rec/rec_" + str(sn) + ".npy"
                rec_true.append(np.load(filename))
                utils.butter_lowpass_filter(rec_true[sn], freq, 1.0/(dt0),order=1)

            fwisolver.freq  = freq
            m0     = np.reshape(vini,-1)
            bounds = [(np.amin(v0),np.amax(v0)) for _ in range(len(m0))]
            result = optimize.minimize(shots, 
                                        m0,
                                        args=(rec_true),
                                        method='L-BFGS-B', 
                                        jac=True, 
                                        tol = 1e-6, 
                                        bounds=bounds, 
                                        options={"disp": True,"eps": 1e-6, "gtol": 1e-6,"maxiter": ite})    
            
            vini   = result.x

    else:
        
        # Reading true data source
        rec_true = []
        
        for sn in range(0, nshots):    
            
            filename = "rec/rec_" + str(sn) + ".npy"
            rec_true.append(np.load(filename))

        fwisolver.freq  = setting["f0"]

        m0     = np.reshape(vini,-1)
        bounds = [(np.amin(v0),np.amax(v0)) for _ in range(len(m0))]
        result = optimize.minimize(shots, 
                                   m0,
                                   args=(rec_true),
                                   method='L-BFGS-B', 
                                   jac=True, 
                                   tol = 1e-6, 
                                   bounds=bounds, 
                                   options={"disp": True,"eps": 1e-6, "gtol": 1e-6,"maxiter": ite})
    #==============================================================================