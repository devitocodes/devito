#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================

#==============================================================================
# Módulos Importados do Python / Devito / Examples
#==============================================================================

#==============================================================================
# Python Modules and Imports
#==============================================================================
import numpy                   as np
import math                    as mt
import sys
import testes_opt              as ttopt
import sympy                   as sym
from   scipy.interpolate       import interp1d 
from   sympy.polys.polyfuncs   import interpolate
#==============================================================================

#==============================================================================
# Basic Coefs
#==============================================================================
sys.path.insert(0, './coef_opt/basic_weights')
import basic_weights                    as coefbasic
#==============================================================================

#==============================================================================
# Square Coefs
#==============================================================================
sys.path.insert(0, './coef_opt/square_weights')
import square_weights                   as coefsquare
#==============================================================================

#==============================================================================
# Square Rhombus Coefs
#==============================================================================
sys.path.insert(0, './coef_opt/square_rhombus_weights')
import square_rhombus_weights           as coefsquarerho
#==============================================================================

#==============================================================================
# Square Rhombus Adpt Coefs
#==============================================================================
sys.path.insert(0, './coef_opt/square_rhombus_adpt_weights')
import square_rhombus_adpt_weights      as coefsquarerhoadpt
#==============================================================================

#==============================================================================
class coefopt1:
#==============================================================================

#==============================================================================    
    def __init__(self,teste):
        self.teste       = teste
        self.hx, self.hz = teste.hx, teste.hz 
        self.sou         = teste.sou
        self.nvalue      = teste.nvalue
        self.exttrap     = teste.exttrap
#==============================================================================

#==============================================================================
    def calccoef(self,wauthor,wtype,sou):
#==============================================================================

#==============================================================================
# T1 - Basic Weights
#==============================================================================
        if(wauthor==0):
            
            nordem = self.sou
            ctex   = 1/(self.hx**2)
            ctez   = 1/(self.hz**2)
#==============================================================================

#==============================================================================
            if(wtype==1):
                                
                Txx, Tshow = coefbasic.calccoef(nordem)
                Tzz, Tshow = coefbasic.calccoef(nordem)
                
                Txx = ctex*Txx
                Tzz = ctez*Tzz
                
                npx = np.size(Txx)
                npz = np.size(Tzz)
                                
                mcoef = np.zeros((npx,npz))
                
                npxm  = int(npx/2)
                
                npzm  = int(npz/2) 
                                
                for i in range(0,npx):
                    
                    mcoef[npxm,i] = mcoef[npxm,i] + Txx[i]
                    
                for j in range(0,npz):
                    
                    mcoef[j,npzm] = mcoef[j,npzm] + Tzz[j]
#==============================================================================

#==============================================================================
# T2 - Square Weights
#==============================================================================
        if(wauthor==1):
            
            nordem           = self.sou
            knumber          = int(nordem/2)
            ctex             = 1/(self.hx**2)
            ctez             = 1/(self.hz**2)
            hval             = min(self.hx,self.hz)
#==============================================================================

#==============================================================================
            if(wtype==1):

                coef_square_type = 0
                square_strategy  = 0                                
                mcoef            = coefsquare.calccoef(knumber,hval,square_strategy,coef_square_type)
                Txx              = 0
                Tzz              = 0
#==============================================================================

#==============================================================================
            if(wtype==2):

                coef_square_type = 1
                square_strategy  = 0                                
                mcoef            = coefsquare.calccoef(knumber,hval,square_strategy,coef_square_type)
                Txx              = 0
                Tzz              = 0
#==============================================================================

#==============================================================================
            if(wtype==3):

                coef_square_type = 1
                square_strategy  = 1                                
                mcoef            = coefsquare.calccoef(knumber,hval,square_strategy,coef_square_type)
                Txx              = 0
                Tzz              = 0
#==============================================================================

#==============================================================================
            if(wtype==4):

                coef_square_type = 1
                square_strategy  = 2                                
                mcoef            = coefsquare.calccoef(knumber,hval,square_strategy,coef_square_type)
                Txx              = 0
                Tzz              = 0
#==============================================================================

#==============================================================================
            if(wtype==5):

                coef_square_type = 1
                square_strategy  = 3                                
                mcoef            = coefsquare.calccoef(knumber,hval,square_strategy,coef_square_type)
                Txx              = 0
                Tzz              = 0
#==============================================================================

#==============================================================================
# T2 - Square Rhombus Weights
#==============================================================================
        if(wauthor==2):
            
            nordem           = self.sou
            mvalue           = int(nordem/2)
            nvalue           = self.nvalue
            ctex             = 1/(self.hx**2)
            ctez             = 1/(self.hz**2)
            hval             = min(self.hx,self.hz)
#==============================================================================

#==============================================================================
            if(wtype==1):
                                
                mcoef = coefsquarerho.calccoef(mvalue,nvalue,hval)
                Txx   = 0
                Tzz   = 0
#==============================================================================

#==============================================================================
# T2 - Square Rhombus Weights
#==============================================================================
        if(wauthor==3):
            
            nordem           = self.sou
            mvalue           = int(nordem/2)
            exttrap          = self.exttrap        
            ctex             = 1/(self.hx**2)
            ctez             = 1/(self.hz**2)
            hval             = min(self.hx,self.hz)
            ctehval          = 1/(hval**2)
#==============================================================================

#==============================================================================
            if(wtype==1):
                                
                mcoef = coefsquarerhoadpt.calccoef(mvalue,exttrap,hval)
                Txx   = 0
                Tzz   = 0
#==============================================================================

#==============================================================================        
        return Txx, Tzz, mcoef
#==============================================================================

#==============================================================================
    def eqconstuct1(self,mcoef,u,x,z):

        npx      = mcoef.shape[0]
        npz      = mcoef.shape[1]
        npxm     = int(npx/2)
        npzm     = int(npz/2)
        initialx = -npxm
        initialz =  npzm
        pdeaux   = 0
        contcoef = 0 
        
        for i in range(0,npx):
            
            for j in range(0,npz):
                                
                a   = int(initialx)
                b   = int(initialz)
                pxs = x + a
                pzs = z + b                
                                
                if(mcoef[i,j]!=0): contcoef = contcoef + 1
                
                pdeaux = pdeaux + u[pxs,pzs]*mcoef[i,j]
                                
                initialx = initialx + 1

            initialx = -npxm
            initialz =  initialz - 1

        return pdeaux, contcoef
#==============================================================================