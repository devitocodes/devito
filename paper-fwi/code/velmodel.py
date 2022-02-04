#==============================================================================
# Python Imports
#==============================================================================
import numpy                   as np
from   scipy.interpolate       import interp1d
from   devito                  import *
#==============================================================================

#==============================================================================
# Marmosi Model
#==============================================================================
def MarmoVelModel(setup,vp,abc):

    compx = setup.compx
    compz = setup.compz
    nptx  = setup.nptx
    nptz  = setup.nptz
    x0    = setup.x0
    x1    = setup.x1
    z0    = setup.z0
    z1    = setup.z1

    nptxvel =  len(vp[:])
    nptzvel =  len(vp[0,:])
    
    x0vel   =  0        
    x1vel   =  17000     
    z0vel   =  0       
    z1vel   =  3500.

    Xvel    = np.linspace(x0vel,x1vel,nptxvel)
    Zvel    = np.linspace(z0vel,z1vel,nptzvel)

    fscale = 10**(-3) 
    vp     = vp*fscale

    X0 = np.linspace(x0,x1,nptx)  
    Z0 = np.linspace(z0,z1,nptz)
    
    C0x = np.zeros((nptx,nptzvel))
    
    for j in range(nptzvel):
        x  = Xvel
        z  = vp[0:nptxvel,j]
        cs = interp1d(x,z,kind='nearest',fill_value="extrapolate")
        xs = X0
        C0x[0:nptx,j] = cs(xs)
    
    v0 = np.zeros((nptx,nptz))  
    
    for i in range(nptx):
        x  = Zvel
        z  = C0x[i,0:nptzvel]
        cs = interp1d(x,z,kind='nearest',fill_value="extrapolate")
        xs = Z0
        v0[i,0:nptz] = cs(xs)

    return v0
#==============================================================================

#==============================================================================
# Circle - Isotropic
#==============================================================================      
def CircleIsot(setup, abcs, r=200, vp_circle=3.0, vp_background=2.5):

    nptx = setup.nptx
    nptz = setup.nptz
    x0    = setup.x0
    x1    = setup.x1
    z0    = setup.z0
    z1    = setup.z1
    x0pml = setup.x0pml
    x1pml = setup.x1pml
    z0pml = setup.z0pml
    z1pml = setup.z1pml
    hx    = setup.hx
    hz    = setup.hz
    npmlx = setup.npmlx
    npmlz = setup.npmlz
    
    xc    = (x1pml-x0pml)/2    
    zc    = (z1pml-z0pml)/2 

    X0   = np.linspace(x0,x1,nptx)
    Z0   = np.linspace(z0,z1,nptz)
    v0   = np.zeros((nptx,nptz))                     
  
    for i in range(nptx):
        for j in range(nptz):
            val = (X0[i]-xc)**2 + (Z0[j]-zc)**2
            if (val<=r**2): 
                v0[i,j] = vp_circle
            else:        
                v0[i,j] = vp_background
    
    return v0                
#==============================================================================

#==============================================================================      
def HorizontalLayers(setup,abcs):

    nptx = setup.nptx
    nptz = setup.nptz
    x0    = setup.x0
    x1    = setup.x1
    z0    = setup.z0
    z1    = setup.z1
    x0pml = setup.x0pml
    x1pml = setup.x1pml
    z0pml = setup.z0pml
    z1pml = setup.z1pml
    hx    = setup.hx
    hz    = setup.hz
    npmlx = setup.npmlx
    npmlz = setup.npmlz
    
    xc    = (x1pml-x0pml)/2    
    zc    = (z1pml-z0pml)/2 

    X0   = np.linspace(x0,x1,nptx)
    Z0   = np.linspace(z0,z1,nptz)
    v0   = np.zeros((nptx,nptz))                     
  
    for i in range(nptx):
        
        for j in range(nptz):
        
            if(Z0[j]<=200):
                
                v0[i,j] = 1.5
            
            elif(Z0[j]>200 and Z0[j]<=500):
            
                v0[i,j] = 3.0
            
            elif(Z0[j]>500 and Z0[j]<=800):
                
                v0[i,j] = 4.0
            
            elif(Z0[j]>800):
            
                v0[i,j] = 5.0
    
    return v0                
#==============================================================================

#==============================================================================
def saltvel(setup,delta,vp0,abc):
   
    compx = setup.compx
    compz = setup.compz
    nptx  = setup.nptx
    nptz  = setup.nptz
    x0    = setup.x0
    x1    = setup.x1
    z0    = setup.z0
    z1    = setup.z1
    
    nptxvel =  len(vp0[:])
    nptzvel =  len(vp0[0,0:175])
    
    x0vel   = 0
    x1vel   = delta*nptxvel
    z0vel   = 0
    z1vel   = delta*nptzvel
    
    Xvel = np.linspace(x0vel,x1vel,nptxvel)
    Zvel = np.linspace(z0vel,z1vel,nptzvel)
    vp   = vp0[:, 0:175]

    X0 = np.linspace(x0,x1,nptx)  
    Z0 = np.linspace(z0,z1,nptz)
    
    C0x = np.zeros((nptx,nptzvel))
    
    for j in range(nptzvel):
        x  = Xvel
        z  = vp[0:nptxvel,j]
        cs = interp1d(x,z,kind='nearest',fill_value="extrapolate")
        xs = X0
        C0x[0:nptx,j] = cs(xs)
    
    v0 = np.zeros((nptx,nptz))  
    
    for i in range(nptx):
        x  = Zvel
        z  = C0x[i,0:nptzvel]
        cs = interp1d(x,z,kind='nearest',fill_value="extrapolate")
        xs = Z0
        v0[i,0:nptz] = cs(xs)

    return v0
#==============================================================================

#==============================================================================
def LinearInitModel(setup, max_vp, min_vp, abc):
    
    nptx  = setup.nptx
    nptz  = setup.nptz
    x0    = setup.x0
    x1    = setup.x1
    z0    = setup.z0
    z1    = setup.z1
    x0pml = setup.x0pml
    x1pml = setup.x1pml
    z0pml = setup.z0pml
    z1pml = setup.z1pml
  
    X0   = np.linspace(x0,x1,nptx)
    Z0   = np.linspace(z0,z1,nptz)

    v0   = np.zeros((nptx,nptz))                     

    xmpml = 0.5*(x0pml+x1pml)
    zmpml = 0.5*(z0pml+z1pml)
        
    pxm = 0
    pzm = 0
        
    for i in range(0,nptx):
        
        if(X0[i]==xmpml): pxm = i
            
    for j in range(0,nptz):
        
        if(Z0[j]==zmpml): pzm = j
            
    p0 = 0    
    p2 = pzm
    p4 = nptz

    b  = min_vp
    a  = (max_vp - b)/(setup.z1pml/1000) 
    
    for i in range(setup.nptz-setup.npmlz):
    
        z = i*setup.hz/1000
        
        v0[:,i] = a*z + b
    
    v0[0:setup.nptx,-setup.npmlz:setup.nptz] = max_vp

    return v0
#==============================================================================

#==============================================================================
# Velocity Model
#==============================================================================
def SetVel(model,setup,setting,grid, **kwargs):
    
    if(model['vp']=='Marmousi'):
        
        vp_file  = kwargs.get('vp_file')
        
        if kwargs.get('start_model') == 'True':
           
            v0       = MarmoVelModel(setup, vp_file, setting["Abcs"])
        
        if kwargs.get('start_model') == 'Initial':
        
            max_vp = 4.5
            min_vp = 1.5
            v0       = LinearInitModel(setup,max_vp,min_vp,setting["Abcs"])
                    
    elif(model['vp']=='Salt'):
        
        vp_file = kwargs.get('vp_file')
        v0      = saltvel(setup, 20,vp_file,setting["Abcs"])
    
    elif(model['vp']=='Circle'):
        
        if kwargs.get('start_model') == 'Initial':
            
            v0 = CircleIsot(setup,setting["Abcs"],vp_circle=2.5)
        
        else:
        
            v0 = CircleIsot(setup,setting["Abcs"])

    elif(model['vp']=='HorizontalLayers'):
        
        if kwargs.get('start_model') == 'Initial':
        
            v0 = CircleIsot(setup,setting["Abcs"],vp_circle=2.5)
        
        else:
        
            v0 = HorizontalLayers(setup,setting["Abcs"]) 
    
    return v0
#==============================================================================