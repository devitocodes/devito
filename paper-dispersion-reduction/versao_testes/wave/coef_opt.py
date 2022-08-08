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
import math                    as mt
import sys
import testes_opt              as ttopt
import macustica               as mc
import sympy                   as sym
from   scipy.interpolate       import interp1d 
from sympy.polys.polyfuncs import interpolate
#==============================================================================
# Devito Imports
#==============================================================================
from devito import *
#==============================================================================
# Devito Examples Imports
#==============================================================================
from   examples.seismic        import TimeAxis
from   examples.seismic        import RickerSource
from   examples.seismic        import Receiver
#==============================================================================

#==============================================================================
# Optimized Coefficients
#==============================================================================
# Yang Liu 2012 Coefficients
#==============================================================================
sys.path.insert(0, './coef_opt/coef_yang_liu')
import coef_otm_versao14        as     cotm14
#==============================================================================
# # Liu-Sen 2009 Coefficients
#==============================================================================
sys.path.insert(0, './coef_opt/coef_liu_sen')
import coef_otm_versao32        as     cotm32
#==============================================================================
# Wang, Liu and Sen - Cross/Rombus Stencils - 2016
#==============================================================================
sys.path.insert(0, './coef_opt/coef_wang_liu_sen')
import coef_otm_versao71        as     cotm71
import coef_otm_versao72        as     cotm72
#==============================================================================

#==============================================================================
class coefopt1:
# Esquemas Low Disperssion para uxx e uyy
#==============================================================================

#==============================================================================    
    def __init__(self,teste,MV):
        self.teste = teste
        self.MV    = mc.acusdevito(self.teste)
        self.hx, self.hy = teste.hx, teste.hy 
        self.t0, self.tn = teste.t0, teste.tn
        self.x0, self.y0 = teste.x0, teste.y0
        self.CFL = teste.CFL
#==============================================================================

#==============================================================================
    def calccoef(self,wauthor,wtype,sou,nvalue,vmax,dt):
#==============================================================================

#==============================================================================
        if(wauthor==1):
                
            nordemliu = int(sou/2)
            ctex      = 1/(self.hx**2)
            ctey      = 1/(self.hy**2)
#==============================================================================

#==============================================================================
# T1 - Yang Liu 2012 Coefficients - FD coefficients calculated by LS
#==============================================================================
            if(wtype==1):

                rvalliu = (vmax*dt)/(min(self.hx,self.hy)) 
                rvalliux = (vmax*dt)/(self.hx)
                rvalliuy = (vmax*dt)/(self.hy)

                if(nordemliu==2):

                    bint = 0.79

                elif(nordemliu==3):

                    bint = 1.23
                
                elif(nordemliu==4):

                    bint = 1.56
                    
                elif(nordemliu==5):

                    bint = 1.82
                
                elif(nordemliu==6):

                    bint = 2.01
                    
                elif(nordemliu==7):

                    bint = 2.16

                elif(nordemliu==8):

                    bint = 2.28

                elif(nordemliu==9):

                    bint = 2.37
                    
                else:

                    bint = 2.45    
                                
                Txx = ctex*cotm11.calccoef(nordemliu,bint,rvalliux)
                Tyy = ctey*cotm11.calccoef(nordemliu,bint,rvalliuy)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
# T2 Type - Coef Yang Liu - A globally optimal FD scheme based on space-domain 
# dispersion relation
#==============================================================================
            if(wtype==2):

                rvalliu = (vmax*dt)/(min(self.hx,self.hy))
                rvalliux = (vmax*dt)/(self.hx)
                rvalliuy = (vmax*dt)/(self.hy)
                
                if(nordemliu==2):
    
                    bint = 0.55

                elif(nordemliu==3):

                    bint = 1.10

                elif(nordemliu==4):

                    bint = 1.53
                    
                elif(nordemliu==5):

                    bint = 1.85
                
                elif(nordemliu==6):

                    bint = 2.08
                    
                elif(nordemliu==7):

                    bint = 2.25
                
                elif(nordemliu==8):

                    bint = 2.38
                
                elif(nordemliu==9):

                    bint = 2.48
                
                else:

                    bint = 2.56    

                Txx = ctex*cotm12.calccoef(nordemliu,bint,rvalliux)
                Tyy = ctey*cotm12.calccoef(nordemliu,bint,rvalliuy)

                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]                
#==============================================================================

#==============================================================================
# T3 Type - Coef Yang Liu - A nearly globally optimal FD scheme based on 1D
# time-space-domain dispersion relation
#==============================================================================
            if(wtype==3):
  
                rvalliu = (vmax*dt)/(min(self.hx,self.hy))
                rvalliux = (vmax*dt)/(self.hx)
                rvalliuy = (vmax*dt)/(self.hy)

                if(nordemliu==2):
    
                    bint = 0.55

                elif(nordemliu==3):

                    bint = 1.10

                elif(nordemliu==4):

                    bint = 1.53
                    
                elif(nordemliu==5):

                    bint = 1.85
                
                elif(nordemliu==6):

                    bint = 2.08
                    
                elif(nordemliu==7):

                    bint = 2.25
                
                elif(nordemliu==8):

                    bint = 2.38
                
                elif(nordemliu==9):

                    bint = 2.48
                
                else:

                    bint = 2.56    
                    
                Txx = ctex*cotm13.calccoef(nordemliu,bint,rvalliux)
                Tyy = ctey*cotm13.calccoef(nordemliu,bint,rvalliuy)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
# T4 Type - Coef Yang Liu - A nearly globally optimal FD scheme based on 2D
# time-space-domain dispersion relation
#==============================================================================
            if(wtype==4):

                rvalliu  = (vmax*dt)/(min(self.hx,self.hy))
                rvalliux = (vmax*dt)/(self.hx)
                rvalliuy = (vmax*dt)/(self.hy)
                
                tetaint = np.pi/4

                if(nordemliu==2):
    
                    bint = 0.55

                elif(nordemliu==3):

                    bint = 1.10

                elif(nordemliu==4):

                    bint = 1.53
                    
                elif(nordemliu==5):

                    bint = 1.85
                
                elif(nordemliu==6):

                    bint = 2.08
                    
                elif(nordemliu==7):

                    bint = 2.25
                
                elif(nordemliu==8):

                    bint = 2.38
                
                elif(nordemliu==9):

                    bint = 2.48
                
                else:

                    bint = 2.56    
                    
                Txx = ctex*cotm14.calccoef(nordemliu,bint,tetaint,rvalliux)
                Tyy = ctey*cotm14.calccoef(nordemliu,bint,tetaint,rvalliuy)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
        if(wauthor==2):            
#==============================================================================
            
#==============================================================================
# T5 Type - Coef Li, Liu, Sen and Ren - A time-space-domain mesh-free finite
# difference based on least squares
#==============================================================================
            if(wtype==1):

                K        = np.pi
                x0       = self.x0
                y0       = self.y0
                hx       = self.hx
                hy       = self.hy
                rvalllsr = dt*vmax
                
                Txx,Tyy  = cotm21.calccoef(sou,K,x0,y0,hx,hy,rvalllsr)
#==============================================================================

#==============================================================================
        if(wauthor==3):      
            
            ctex = 1/(self.hx**2)
            ctey = 1/(self.hy**2) 
#==============================================================================
            
#==============================================================================
# T6 Type - Coef Liu and Sen - Time–space domain high-order FD method for 1D 
# acoustic  wave equations 
#==============================================================================
            if(wtype==1):
                
                rvalx = dt*vmax/self.hx
                rvaly = dt*vmax/self.hy
                Txx   = ctex*cotm31.calccoef(sou,rvalx)
                Tyy   = ctey*cotm31.calccoef(sou,rvaly)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
# T7 Type - Coef Liu and Sen - Time–space domain high-order FD method for 2D 
# acoustic  wave equations
#==============================================================================
            if(wtype==2):

                rval    = dt*vmax/min(self.hx,self.hy)
                rvalx   = (vmax*dt)/(self.hx)
                rvaly   = (vmax*dt)/(self.hy)
                
                tetaval = np.pi/8
                   
                Txx = ctex*cotm32.calccoef(sou,rvalx,tetaval)
                Tyy = ctey*cotm32.calccoef(sou,rvaly,tetaval)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
        if(wauthor==4):
                        
            nordem = int(sou/2) 
#==============================================================================
            
#==============================================================================
# T8 Type- Coef Yajun - Uniform dispersion reduction schemes
#==============================================================================
            if(wtype==1):

                rvalx   = dt*vmax/self.hx
                rvaly   = dt*vmax/self.hy
                knumber = 2
                
                Txx = cotm41.calccoef(nordem,rvalx,knumber)
                Tyy = cotm41.calccoef(nordem,rvaly,knumber)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
# T8 Type- Coef Yajun - Uniform dispersion reduction schemes
#==============================================================================
            if(wtype==2):

                rvalx    = dt*vmax/self.hx
                rvaly    = dt*vmax/self.hy
                knumber  = 2
                phiangle = np.pi/4
                
                Txx = cotm42.calccoef(nordem,rvalx,knumber,phiangle)
                Tyy = cotm42.calccoef(nordem,rvaly,knumber,phiangle)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
        if(wauthor==5):
            
            nordem = int(sou/2)    
#==============================================================================
            
#==============================================================================
# T9 Type - Coef Kastner e Finkelstein - 2006 and 2007
#==============================================================================
            if(wtype==1):
               
                rvalx   = dt*vmax/self.hx
                rvaly   = dt*vmax/self.hy
                knumber = 2

                Txx = cotm51.calccoef(nordem,rvalx,knumber)
                Tyy = cotm51.calccoef(nordem,rvaly,knumber)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
# T10 Type - Coef Kastner e Finkelstein - 2006 and 2007
#==============================================================================
            if(wtype==2):
               
                rvalx   = dt*vmax/self.hx
                rvaly   = dt*vmax/self.hy
                knumber = 2

                Txx = cotm52.calccoef(nordem,rvalx,knumber)
                Tyy = cotm52.calccoef(nordem,rvaly,knumber)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
        if(wauthor==6):
            
            ctex  = 1/(self.hx**2)
            ctey  = 1/(self.hy**2)
#==============================================================================
            
#==============================================================================
# T11 Type - Edward Caunt - Master Dissertations Coefficients
#==============================================================================
            if(wtype==1):
               
                ndim = 1 
                Txx  = ctex*cotm61.calccoef(sou,ndim)
                Tyy  = ctey*cotm61.calccoef(sou,ndim)
                
                npx = np.size(Txx)
                npy = np.size(Tyy)
                                
                mcoef = np.zeros((npx,npy))
                
                npxm  = int(npx/2)
                
                npym  = int(npy/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npy):
                    
                    mcoef[j,npym] = mcoef[j,npym] + Tyy[j]
#==============================================================================

#==============================================================================
        if(wauthor==7):
                        
            mvalue  = int(sou/2)
            nvalue  = nvalue
            if(nvalue>mvalue): sys.exit('N>M')
            rvalx   = dt*vmax/self.hx
            rvaly   = dt*vmax/self.hy
            ctex    = 1/(self.hx**2)
            ctey    = 1/(self.hy**2)
#==============================================================================
            
#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
            if(wtype==1):
               
                Txx    = 0.0
                Tyy    = 0.0
                cte    = min(ctex,ctey)
                mcoef  = cte*cotm71.calccoef(mvalue,nvalue,rvalx,rvaly)            
#==============================================================================

#==============================================================================
# T13 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
            if(wtype==2):
               
                Txx    = 0.0
                Tyy    = 0.0
                cte    = min(ctex,ctey)
                mcoef  = cte*cotm72.calccoef(mvalue,nvalue,rvalx,rvaly)            
#==============================================================================

#==============================================================================        
        return Txx, Tyy, mcoef
#==============================================================================

#==============================================================================
    def eqconstuct1(self,mcoef,u,t,x,y):
        
        npx      = mcoef.shape[0]
        npy      = mcoef.shape[1]
        npxm     = int(npx/2)
        npym     = int(npy/2)
        initialx = -npxm
        initialy =  npym
        pdeaux   = 0
        contcoef = 0 
        
        for i in range(0,npx):
            
            for j in range(0,npy):
                                
                a   = int(initialx)
                b   = int(initialy)
                pxs = x + a
                pys = y + b                
                                
                if(mcoef[i,j]!=0): contcoef = contcoef + 1
                
                pdeaux = pdeaux + u[t,pxs,pys]*mcoef[i,j]
                                
                initialx = initialx + 1

            initialx = -npxm
            initialy =  initialy - 1

        return pdeaux, contcoef
#==============================================================================

#==============================================================================
    def eqconstuct2(self,mcoefs,u,t,x,y,vrange,vmax,vel):
        
        npx       = mcoefs.shape[1]
        npy       = mcoefs.shape[2]
        npxm      = int(npx/2)
        npym      = int(npy/2)
        initialx  = -npxm
        initialy  =  npym
        pdeaux    = 0
        contcoef  = 0 
        mcoef_new = np.zeros((mcoefs.shape[1],mcoefs.shape[2]))

        for i in range(0,npx):
            
            for j in range(0,npy):
                                
                a   = int(initialx)
                b   = int(initialy)
                pxs = x + a
                pys = y + b                
                
                xvet             = vrange[:]
                yvet             = mcoefs[:,i,j]
                cs               = interp1d(xvet,yvet,kind='nearest',fill_value="extrapolate")
                xs               = vmax
                new_weight       = cs(vmax)               
                mcoef_new[i,j]   = new_weight

                if(mcoef_new[i,j]!=0): contcoef = contcoef + 1
                
                pdeaux = pdeaux + u[t,pxs,pys]*mcoef_new[i,j]
                                
                initialx = initialx + 1

            initialx = -npxm
            initialy =  initialy - 1

        return pdeaux, contcoef, mcoef_new
#==============================================================================

#==============================================================================
    def eqconstuct3(self,mcoefs,u,t,x,y,vel,vrange):
        
        npx       = mcoefs.shape[1]
        npy       = mcoefs.shape[2]
        npxm      = int(npx/2)
        npym      = int(npy/2)
        initialx  = -npxm
        initialy  =  npym
        pdeaux    = 0
        contcoef  = 0 

        for i in range(0,npx):
            
            for j in range(0,npy):
                                
                a   = int(initialx)
                b   = int(initialy)
                pxs = x + a
                pys = y + b
                
                list_points = list()

                for m3 in range(0,vrange.shape[0]):
                    
                    pair = (vrange[m3],mcoefs[m3,i,j])
                    list_points.append(pair)
                
                #weight_interp    = interpolate(mcoefs[:,i,j],vel)
                weight_interp    = interpolate(list_points,vel)
                pdeaux           = pdeaux + u[t,pxs,pys]*weight_interp
                initialx         = initialx + 1

            initialx = -npxm
            initialy =  initialy - 1

        return pdeaux
#==============================================================================