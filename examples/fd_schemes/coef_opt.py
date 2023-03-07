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
from   sympy.polys.polyfuncs   import interpolate
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
import coef_otm_versao14        as     cotm11
#==============================================================================
# # Liu-Sen 2009 Coefficients
#==============================================================================
sys.path.insert(0, './coef_opt/coef_liu_sen')
import coef_otm_versao22        as     cotm21
#==============================================================================
# Wang, Liu and Sen - Cross/Rombus Stencils - 2016
#==============================================================================
sys.path.insert(0, './coef_opt/coef_wang_liu_sen')
import coef_otm_versao31        as     cotm31
import coef_otm_versao32        as     cotm32
#==============================================================================
# Square Rhombus Adpt Coefs
#==============================================================================
sys.path.insert(0, './coef_opt/coef_cross_square')
import coef_otm_versao41        as     cotm41
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
        self.exttrap = teste.exttrap
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
# Coef Yang Liu - A nearly globally optimal FD scheme based on 2D
# time-space-domain dispersion relation
#==============================================================================
            if(wtype==1):

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
                    
                Txx = ctex*cotm11.calccoef(nordemliu,bint,tetaint,rvalliux)
                Tyy = ctey*cotm11.calccoef(nordemliu,bint,tetaint,rvalliuy)
                
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
            
            ctex = 1/(self.hx**2)
            ctey = 1/(self.hy**2) 
#==============================================================================
            
#==============================================================================
# Coef Liu and Sen - Time–space domain high-order FD method for 2D 
# acoustic  wave equations
#==============================================================================
            if(wtype==1):

                rval    = dt*vmax/min(self.hx,self.hy)
                rvalx   = (vmax*dt)/(self.hx)
                rvaly   = (vmax*dt)/(self.hy)
                
                tetaval = np.pi/8
                   
                Txx = ctex*cotm21.calccoef(sou,rvalx,tetaval)
                Tyy = ctey*cotm21.calccoef(sou,rvaly,tetaval)
                
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
        if(wauthor==3):
                        
            mvalue  = int(sou/2)
            nvalue  = nvalue
            if(nvalue>mvalue): sys.exit('N>M')
            rvalx   = dt*vmax/self.hx
            rvaly   = dt*vmax/self.hy
            ctex    = 1/(self.hx**2)
            ctey    = 1/(self.hy**2)
#==============================================================================
            
#==============================================================================
# Coef Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
            if(wtype==1):
               
                Txx    = 0.0
                Tyy    = 0.0
                cte    = min(ctex,ctey)
                mcoef  = cte*cotm31.calccoef(mvalue,nvalue,rvalx,rvaly)            
#==============================================================================

#==============================================================================
# Coef Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
            if(wtype==2):
               
                Txx    = 0.0
                Tyy    = 0.0
                cte    = min(ctex,ctey)
                mcoef  = cte*cotm32.calccoef(mvalue,nvalue,rvalx,rvaly)            
#==============================================================================

#==============================================================================
# Coef Cross Square
#==============================================================================
        if(wauthor==4):
            
            mvalue           = int(sou/2)
            exttrap          = self.exttrap        
            ctex             = 1/(self.hx**2)
            ctez             = 1/(self.hy**2)
            hval             = min(self.hx,self.hy)
            ctehval          = 1/(hval**2)
#==============================================================================

#==============================================================================
            if(wtype==1):
                                
                mcoef = cotm41.calccoef(mvalue,exttrap,hval)
                Txx   = 0
                Tyy   = 0
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
