#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================

#==============================================================================
# Módulos Importados do Python / Devito / Examples
#==============================================================================

#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                 as np
import math                  as mt
import segyio
from   scipy.interpolate     import interp1d    
#==============================================================================

#==============================================================================
class acusdevito:
#==============================================================================

#==============================================================================    
    def __init__(self,teste):
        self.teste = teste
        #self.C0a   = self.geramvel(teste)
        #self.C0a   = self.geramvel2(teste)
        #self.C0a   = self.geramvel3(teste)
        #self.C0a   = self.geramvel4(teste)
        #self.C0a   = self.geramvel7(teste)
        #self.C0a   = self.geramvel8(teste)
        self.C0a   = self.geramvel9(teste)
#==============================================================================

#==============================================================================        
    def geramvel(self,teste):
    
        nptx      = teste.nptx
        npty      = teste.npty
        C0a       = np.zeros((nptx,npty))                     
        C0a[:,:]  = 4.0
                        
        return C0a
#==============================================================================

#==============================================================================        
    def geramvel2(self,teste):
    
        nptx      = teste.nptx
        npty      = teste.npty
        
        C0a       = np.zeros((nptx,npty))                     
      
        p0 = 0    
        p2 = int(0.6*npty)
        p4 = npty
                
        C0a[0:nptx,p0:p2] = 2.0
        C0a[0:nptx,p2:p4] = 4.0
                        
        return C0a
#==============================================================================

#==============================================================================        
    def geramvel3(self,teste):
    
        nptx      = teste.nptx
        npty      = teste.npty
        X0        = teste.X0
        Y0        = teste.Y0
        datainter = teste.datainter
        
        nptxvel = 677
        nptyvel = 211
        x0vel   = 0.
        x1vel   = 13520.     
        y0vel   = 0.        
        y1vel   = 4200.      
        hxvel   = (x1vel-x0vel)/(nptxvel-1)
        hyvel   = (y1vel-y0vel)/(nptyvel-1)
        Xvel    = np.linspace(x0vel,x1vel,nptxvel)
        Yvel    = np.linspace(y0vel,y1vel,nptyvel)

        velold  = np.load('dados_velocidade/seg_eage_xcut_338.npy')

        C0        = np.zeros((nptx,npty))                         
        
        veltemp1 = np.zeros((nptxvel,nptyvel))
        veltemp1[0:nptxvel-1,0:nptyvel-1] = velold[:,:]
        veltemp1[nptxvel-1:,0:nptyvel]    = veltemp1[nptxvel-2,0:nptyvel]
        veltemp1[0:nptxvel:,nptyvel-1]    = veltemp1[0:nptxvel,nptyvel-2]
        
        veltemp2 = np.zeros((nptxvel,nptyvel))
        nycorte1 = 0   
        nycorte2 = 170
        ndif1    = nptyvel - nycorte2
        veltemp2[0:nptxvel,ndif1:nptyvel] = veltemp1[0:nptxvel,nycorte1:nycorte2]
        
        for j in range(0,ndif1):
        
            veltemp2[0:nptxvel,j] = np.min(veltemp1)

        vel = np.zeros((nptxvel,nptyvel))
        vel = veltemp2
                         
        if(datainter==0):
            
            C0 = vel
        
        else:
        
            C0x = np.zeros((nptx,nptyvel))
            for j in range(nptyvel):
                x = Xvel
                y = vel[0:nptxvel,j]
                cs = interp1d(x,y,kind='nearest',fill_value="extrapolate")
                xs = X0
                C0x[0:nptx,j] = cs(xs)
            for i in range(nptx):
                x = Yvel
                y = C0x[i,0:nptyvel]
                cs = interp1d(x,y,kind='nearest',fill_value="extrapolate")
                xs = Y0
                C0[i,0:npty] = cs(xs)
        
        # Manufacturing Field
        C0        = np.around(C0,2)
        vlimit    = np.amax(C0)
        new_limit = 5.0
        C0        = np.where(C0>=vlimit,new_limit,C0)
    
        return C0
#==============================================================================

#==============================================================================        
    def geramvel4(self,teste):
    
        nptx      = teste.nptx
        npty      = teste.npty
        X0        = teste.X0
        Y0        = teste.Y0
        datainter = teste.datainter
        fscale    = 10**(-3) 
        C0        = np.zeros((nptx,npty))                     
    
        nptxvel = 1407
        nptyvel = 311
        x0vel   = 0.        
        x1vel   = 70300.     
        y0vel   = 0.        
        y1vel   = 9920.
        hxvel   = (x1vel-x0vel)/(nptxvel-1)
        hyvel   = (y1vel-y0vel)/(nptyvel-1)
        Xvel    = np.linspace(x0vel,x1vel,nptxvel)
        Yvel    = np.linspace(y0vel,y1vel,nptyvel)
        vel     = np.fromfile('dados_velocidade/gatoDoMato2DCut.bin',dtype='float32')
        vel     = vel.reshape((nptxvel,nptyvel))
        vel     = fscale*vel
        
        if(datainter==0):
            
            C0 = vel
        
        else:
        
            C0x = np.zeros((nptx,nptyvel))
            for j in range(nptyvel):
                x = Xvel
                y = vel[0:nptxvel,j]
                cs = interp1d(x,y,kind='nearest',fill_value="extrapolate")
                xs = X0
                C0x[0:nptx,j] = cs(xs)
            for i in range(nptx):
                x = Yvel
                y = C0x[i,0:nptyvel]
                cs = interp1d(x,y,kind='nearest',fill_value="extrapolate")
                xs = Y0
                C0[i,0:npty] = cs(xs)
        
        # Manufacturing Field
        C0        = np.around(C0,2)
        vlimit    = np.amax(C0)
        new_limit = 8.0
        C0        = np.where(C0>=vlimit,new_limit,C0)

        return C0
#==============================================================================

#==============================================================================        
    def geramvel7(self,teste):
    
        nptx      = teste.nptx
        npty      = teste.npty
        C0a       = np.zeros((nptx,npty))                     
        C0a[:,:]  = 3.0
                        
        return C0a
#==============================================================================

#==============================================================================        
    def geramvel8(self,teste):
    
        nptx      = teste.nptx
        npty      = teste.npty
        Y0        = teste.Y0
        C0a       = np.zeros((nptx,npty))                     
        
        p0 = 0    
        
        py1 = 1200
        
        for i in range(0,Y0.shape[0]):
            
            if(Y0[i]==py1): p2 = i
            
        p4 = npty
                
        C0a[0:nptx,p0:p2] = 1.5
        C0a[0:nptx,p2:p4] = 3.0
                        
        return C0a
#==============================================================================

#==============================================================================        
    def geramvel9(self,teste):
    
        nptx      = teste.nptx
        npty      = teste.npty
        Y0        = teste.Y0
        C0a       = np.zeros((nptx,npty))                     
        
        p0 = 0    
        
        py1 = 1200
        
        for i in range(0,Y0.shape[0]):
            
            if(Y0[i]==py1): p2 = i
            
        p4 = npty
                
        C0a[0:nptx,p0:p2] = 2.0
        C0a[0:nptx,p2:p4] = 8.0
                        
        return C0a
#==============================================================================