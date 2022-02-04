#==============================================================================
# Python Imports
#==============================================================================
import numpy                   as np
import math                    as mt
import os
import pywt
from   scipy.interpolate       import interp1d
from   distributed             import Client, wait
from   scipy.signal            import butter, filtfilt, sosfilt
#==============================================================================

#==============================================================================
# Devito Imports
#==============================================================================
from   devito import *
from   examples.seismic        import TimeAxis
#==============================================================================

#==============================================================================
class ProblemSetup:
#==============================================================================

#==============================================================================
    def __init__(self, setting):
        
        self.set    = setting
        self.lx     = setting["lenpmlx"]  
        self.lz     = setting["lenpmlz"]    
        self.x0     = setting["x0"] - self.lx                          
        self.x1     = setting["lenx"] + setting["x0"] + self.lx        
        self.z0     = setting["z0"]                                    
        self.z1     = setting["lenz"] + setting["z0"] + self.lz        
        self.hx     = setting["lenx"]/(setting["nptx"]-1)              
        self.hz     = setting["lenz"]/(setting["nptz"]-1)
        self.npmlx  = int(self.lx /self.hx)                            
        self.npmlz  = int(self.lz /self.hz)                            
        self.compx  = self.x1-self.x0
        self.nptx   = setting["nptx"]  + 2*self.npmlx 
        self.compz  = self.z1 - self.z0
        self.nptz   = setting["nptz"]  + self.npmlz 
        self.x0pml  = self.x0 + self.npmlx*self.hx                     
        self.x1pml  = self.x1 - self.npmlx*self.hx                     
        self.z0pml  = self.z0                                          
        self.z1pml  = self.z1 - self.npmlz*self.hz                     
        self.tou        = 2                                            
        self.sou        = 12                                           
        self.tn         = self.set["tn"]
        self.t0         = self.set["t0"]  
        self.cfl        = 0.5 
        self.f0         = self.set["f0"]  
        self.Abcs       = self.set["Abcs"]  
        self.shotposition_z = self.set["shotposition_z"]
        self.recposition_z  = self.set["recposition_z"] 
        self.rec_n = self.set["rec_n"] 
        self.habcw = self.set["habcw"] 
#==============================================================================

#==============================================================================  
    def TimeDiscret(self, v0):

        cfl = self.cfl

        tau   = self.tn-self.t0              
        vmax  = np.amax(v0[self.npmlx:-self.npmlx,0:-self.npmlz]) 
        dtmax = np.float64((min(self.hx,self.hz)*cfl)/(vmax))
        ntmax = int((tau)/dtmax)+1
        dt0   = 1.0
        time_range = TimeAxis(start=self.t0,stop=self.tn,step=dt0)
        nt         = time_range.num - 1 

        return dt0, nt, time_range     
#==============================================================================
                                        
#==============================================================================
def makeweights(setup, habcw):
    
    nptx  = setup.nptx
    nptz  = setup.nptz
    npmlx = setup.npmlx
    npmlz = setup.npmlz
  
    pesosx     = np.zeros(npmlx)
    pesosz     = np.zeros(npmlz)
    Mweightsx  = np.zeros((nptx,nptz))
    Mweightsz  = np.zeros((nptx,nptz))

    if(habcw==1):
    
        for i in range(0,npmlx):
            
            pesosx[i] = (npmlx-i)/(npmlx)
    
        for i in range(0,npmlz):
            
            pesosz[i] = (npmlz-i)/(npmlz)
    
    if(habcw==2):
        
        mx = 2
        mz = 2
        
        alphax  = 1.5 + 0.07*(npmlx-mx)    
        alphaz  = 1.5 + 0.07*(npmlz-mz)
            
        for i in range(0,npmlx):
                
            if(0<=i<=(mx)):
                
                pesosx[i] = 1
            
            elif((mx+1)<=i<=npmlx-1):
            
                pesosx[i] = ((npmlx-i)/(npmlx-mx))**(alphax)
            else:
             
                pesosx[i] = 0
    
        for i in range(0,npmlz):
                
            if(0<=i<=(mz)):
                
                pesosz[i] = 1
            
            elif((mz+1)<=i<=npmlz-1):
                
                pesosz[i] = ((npmlz-i)/(npmlz-mz))**(alphaz)
            
            else:
                
                pesosz[i] = 0
      
    for k in range(0,npmlx):
            
        ai = k
        af = nptx - k - 1 
        bi = 0
        bf = nptz - k
        Mweightsx[ai,bi:bf] = pesosx[k]
        Mweightsx[af,bi:bf] = pesosx[k]
                        
    for k in range(0,npmlz):
            
        ai = k
        af = nptx - k 
        bf = nptz - k - 1        
        Mweightsz[ai:af,bf] = pesosz[k]
        
    return Mweightsx,Mweightsz
#==============================================================================

#==============================================================================
def fdamp(x,z,setup,v0, abcs, **kwargs):
   
    x0pml  = setup.x0pml
    x1pml  = setup.x1pml
    z0pml  = setup.z0pml            
    z1pml  = setup.z1pml

    if(abcs=='damping'):
       
        quibar  = 1.5*np.log(1.0/0.001)/(40)
        cte     = 1./np.amax(v0) 
       
        a = np.where(x<=x0pml,(np.abs(x-x0pml)/setup.lx),np.where(x>=x1pml,(np.abs(x-x1pml)/setup.lx),0.))
        b = np.where(z<=z0pml,(np.abs(z-z0pml)/setup.lz),np.where(z>=z1pml,(np.abs(z-z1pml)/setup.lz),0.))
        
        adamp = quibar*(a-(1./(2.*np.pi))*np.sin(2.*np.pi*a))/setup.hx
        bdamp = quibar*(b-(1./(2.*np.pi))*np.sin(2.*np.pi*b))/setup.hz
        
        fdamp = cte*(adamp+bdamp)
    
    elif(abcs=='pml'): 
    
        i=kwargs.get('i')

        ctex = (-3/2)*(1/setup.lx)*np.amax(v0)*np.log(0.001)
        ctez = (-3/2)*(1/setup.lz)*np.amax(v0)*np.log(0.001)
        quibarx  = ctex
        quibarz  = ctez

        quibar  = 0.05
        
        if(i==1):
            
            a = np.where(x<=x0pml,(np.abs(x-x0pml)/setup.lx),np.where(x>=x1pml,(np.abs(x-x1pml)/setup.lx),0.))
            #fdamp = quibar*(a-(1./(2.*np.pi))*np.sin(2.*np.pi*a))
            fdamp = quibarx*(a-(1./(2.*np.pi))*np.sin(2.*np.pi*a))
        
        if(i==2):
        
            a = np.where(z<=z0pml,(np.abs(z-z0pml)/setup.lz),np.where(z>=z1pml,(np.abs(z-z1pml)/setup.lz),0.))
            #fdamp = quibar*(a-(1./(2.*np.pi))*np.sin(2.*np.pi*a))
            fdamp = quibarz*(a-(1./(2.*np.pi))*np.sin(2.*np.pi*a))
   
    elif(abcs=='cpml'): 
    
        i=kwargs.get('i')

        quibar  = 0.1
        
        ctex = (-3/2)*(1/setup.lx)*np.amax(v0)*np.log(0.001)
        ctez = (-3/2)*(1/setup.lz)*np.amax(v0)*np.log(0.001)
        quibarx  = ctex
        quibarz  = ctez

        
        if(i==1):
            
            a = np.where(x<=x0pml,(np.abs(x-x0pml)/setup.lx),np.where(x>=x1pml,(np.abs(x-x1pml)/setup.lx),0.))
            #fdamp = quibar*(a**2)
            fdamp = quibarx*(a**2)
        
        if(i==2):
        
            a = np.where(z<=z0pml,(np.abs(z-z0pml)/setup.lz),np.where(z>=z1pml,(np.abs(z-z1pml)/setup.lz),0.))
            #fdamp = quibar*(a**2)
            fdamp = quibarz*(a**2)
    else:
        
        assert "Invalid option"

    return fdamp
#==============================================================================

#==============================================================================
def makemdamp(setup,v0,abcs):

    x1     = setup.x1
    z1     = setup.z1
    X0     = np.linspace(setup.x0,x1,setup.nptx)    
    Z0     = np.linspace(setup.z0,z1,setup.nptz)  

    if(abcs=='damping'):
       
        X0grid,Z0grid = np.meshgrid(X0,Z0)  
        D0 = np.zeros((setup.nptx,setup.nptz))         
        D0 = np.transpose(fdamp(X0grid,Z0grid, setup,v0, abcs))
    
        return D0
    
    elif(abcs=='pml'):
        
        nptx = setup.nptx
        nptz = setup.nptz
        
        X0   = np.linspace(setup.x0,x1,nptx)    
        Z0   = np.linspace(setup.z0,z1,nptz)
        X0grid,Z0grid = np.meshgrid(X0,Z0)
        
        X1   = np.linspace((setup.x0+0.5*setup.hx),(x1-0.5*setup.hx),nptx-1)
        Z1   = np.linspace((setup.z0+0.5*setup.hz),(z1-0.5*setup.hz),nptz-1)
        X1grid,Z1grid = np.meshgrid(X1,Z1)
    
        D01 = np.zeros((nptx,nptz))
        D02 = np.zeros((nptx,nptz))
        D11 = np.zeros((nptx,nptz))
        D12 = np.zeros((nptx,nptz))
        
        D01 = np.transpose(fdamp(X0grid,Z0grid, setup,v0, abcs,i=1))
        D02 = np.transpose(fdamp(X0grid,Z0grid, setup,v0, abcs,i=2))
    
        D11 = np.transpose(fdamp(X1grid,Z1grid, setup,v0, abcs,i=1))
        D12 = np.transpose(fdamp(X1grid,Z1grid, setup,v0, abcs,i=2))

        return D01, D02, D11, D12    
    
    elif(abcs=='cpml'):
        
        nptx = setup.nptx
        nptz = setup.nptz
        
        X0   = np.linspace(setup.x0,x1,nptx)    
        Z0   = np.linspace(setup.z0,z1,nptz)
        X0grid,Z0grid = np.meshgrid(X0,Z0)
           
        D01 = np.zeros((nptx,nptz))
        D02 = np.zeros((nptx,nptz))

        D01 = np.transpose(fdamp(X0grid,Z0grid, setup,v0, abcs,i=1))
        D02 = np.transpose(fdamp(X0grid,Z0grid, setup,v0, abcs,i=2))
    
        return D01, D02  
#==============================================================================

#==============================================================================        
def makeweightscpml(setup, v0, makemdamp, dt0):
    
    D01,D02 = makemdamp
    nptx    = setup.nptx
    nptz    = setup.nptz
    f0      = setup.f0
    x0pml   = setup.x0pml   
    x1pml   = setup.x1pml   
    z0pml   = setup.z0pml   
    y1pml   = setup.z1pml   
    X0      = np.linspace(setup.x0,setup.x1,nptx)   
    Z0      = np.linspace(setup.z0,setup.z1,nptz)    
    deltax  = setup.compx  
    deltaz  = setup.compz      
    
    A1C = np.zeros((nptx,nptz))
    A2C = np.zeros((nptx,nptz))
    B1C = np.zeros((nptx,nptz))
    B2C = np.zeros((nptx,nptz))

    alpha1max = np.pi*(f0)
    alpha2max = np.pi*(f0)
            
    alpha1v = np.zeros((nptx,nptz))
    alpha2v = np.zeros((nptx,nptz))    
    
    mpower = 1
    
    for i in range(0,nptx):
        #alpha1v[i,:] = np.where(X0[i]<=x0pml,(alpha1max*np.abs(X0[i]-x0pml+deltax)/deltax),np.where(X0[i]>=x1pml,(alpha1max*np.abs(X0[i]-x1pml-deltax)/deltax),alpha1max))
        alpha1v[i,:] = np.where(X0[i]<=x0pml,(alpha1max*(np.abs(X0[i]-x0pml+deltax)/deltax)**mpower),np.where(X0[i]>=x1pml,(alpha1max*(np.abs(X0[i]-x1pml-deltax)/deltax)**mpower),alpha1max))

    for j in range(0,nptz):
        #alpha2v[:,j] = np.where(Z0[j]<=z0pml,(alpha2max*np.abs(Z0[j]-z0pml+deltaz)/deltaz),np.where(Z0[j]>=y1pml,(alpha2max*np.abs(Z0[j]-y1pml-deltaz)/deltaz),alpha2max))
        alpha2v[:,j] = np.where(Z0[j]<=z0pml,(alpha2max*(np.abs(Z0[j]-z0pml+deltaz)/deltaz)**mpower),np.where(Z0[j]>=y1pml,(alpha2max*(np.abs(Z0[j]-y1pml-deltaz)/deltaz)**mpower),alpha2max))
        
    A1C[0:nptx,0:nptz] = np.exp(-(D01[0:nptx,0:nptz]+alpha1v[0:nptx,0:nptz])*dt0)
    A2C[0:nptx,0:nptz] = np.exp(-(D02[0:nptx,0:nptz]+alpha2v[0:nptx,0:nptz])*dt0)
    B1C[0:nptx,0:nptz] = (D01[0:nptx,0:nptz]*(A1C[0:nptx,0:nptz]-1))/(D01[0:nptx,0:nptz]+alpha1v[0:nptx,0:nptz])
    B2C[0:nptx,0:nptz] = (D02[0:nptx,0:nptz]*(A2C[0:nptx,0:nptz]-1))/(D02[0:nptx,0:nptz]+alpha2v[0:nptx,0:nptz])
            
    return A1C,A2C,B1C,B2C,alpha1v,alpha2v
#==============================================================================

#==============================================================================
def makev1m0(setup,v0):

    compx = setup.compx
    compz = setup.compz
    nptx  = setup.nptx
    nptz  = setup.nptz
    x0    = setup.x0
    x1    = setup.x1
    z0    = setup.z0
    z1    = setup.z1
    hx    = setup.hx
    hz    = setup.hz

    X0 = np.linspace(x0,x1,nptx)  
    Z0 = np.linspace(z0,z1,nptz)

    X1 = np.linspace((x0+0.5*setup.hx),(x1-0.5*setup.hx),nptx-1)
    Z1 = np.linspace((z0+0.5*setup.hz),(z1-0.5*setup.hz),nptz-1)    

    v1 = np.zeros((nptx-1,nptz-1))
    C11x = np.zeros((nptx-1,nptz))

    for j in range(nptz):
        x = X0
        z = v0[0:nptx,j]
        cs = interp1d(x,z,kind='nearest',fill_value="extrapolate")
        xs = X1
        C11x[0:nptx-1,j] = cs(xs)
            
    for i in range(nptx-1):
        x  = Z0
        z  = C11x[i,0:nptz]
        cs = interp1d(x,z,kind='nearest',fill_value="extrapolate")
        xs = Z1
        v1[i,0:nptz-1] = cs(xs)

    return v1      
#==============================================================================

#==============================================================================
def butter_lowpass_filter(shot, cutoff, fs, order=1):
    
    """ Low-pass filter the shot record with sampling-rate fs Hz
        and cutoff freq. Hz
    """
    
    nyq = 0.5*fs*1000  # Nyquist Frequency
    normal_cutoff = (cutoff) / nyq
  
    # Get the filter coefficients  
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    
    nc, nr = np.shape(shot)

    for rec in range(nr):
        
        shot[:,rec] = filtfilt(b, a, shot[:,rec])
#==============================================================================

#==============================================================================
def wavelet(rec, n, w, keep):
    
    '''
    Input parameters:
        recs: number os shots
        n: level of decomposition
        w: wavelet family
        keep: percentage of coefficients to keep (0.2 gives 80% of compression)
        input_path: seismogram path
        output_path: path to save wavelets   
    '''

    coeffs = pywt.wavedec2(rec,wavelet=w,level=n)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind # Threshold small indices
    
    coeffs_filt = pywt.array_to_coeffs(Cfilt,coeff_slices,output_format='wavedec2')
    Arecon = pywt.waverec2(coeffs_filt,wavelet=w)
    samples = rec.shape[0]
    columns = rec.shape[1]
    Hsub = Arecon[:samples,:columns]
    
    return Hsub
#==============================================================================