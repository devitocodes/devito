#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================

#==============================================================================
# Módulos Importados do Python / Devito / Examples
#==============================================================================

#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
from   sympy import *
#==============================================================================

#==============================================================================
class ellipctdevito:
#==============================================================================

#==============================================================================    
    def __init__(self,teste):
        self.teste     = teste
        self.F        = self.f(teste)
        self.F_ext    = self.f_ext(teste)
        self.DF2exact = self.df2(teste)
#==============================================================================

#==============================================================================
    def f(self,teste):
    
        teste_type   = teste.teste_type
        X            = teste.X
        Z            = teste.Z
        nptx         = teste.nptx
        nptz         = teste.nptz
        x,z          = symbols('x z', real=True)
        Xgrid, Zgrid = np.meshgrid(X,Z)
        F            = np.zeros((nptx,nptz))
    
        if(teste_type==1):
    
            f       = sin(pi*x)*sin(pi*z)
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgrid,Zgrid))
                  
            #F[:,:] = np.sin(np.pi*Xgrid)*np.sin(np.pi*Zgrid)
        
        if(teste_type==2):
    
            f       = x**4 + z**4 
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgrid,Zgrid))
            
            #F[:,:] = Xgrid**4 + Zgrid**4
   
        if(teste_type==3):
            
            factor  = 1000   
            f       = sin(factor*x)*sin(factor*z)
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgrid,Zgrid))
            
            #factor = 1000   
            #F[:,:]      = np.sin(factor*Xgrid)*np.sin(factor*Zgrid)
   
        if(teste_type==4):

            f       = x**2 + z**2
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgrid,Zgrid))
            
            #F[:,:] = Xgrid**2 + Zgrid**2

        if(teste_type==5):

            a       = 1000
            b       = 3000
            f       = exp(-(a*x**2+b*z**2)) 
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgrid,Zgrid))
            
            #a = 1000
            #b = 3000
            #F[:,:] = np.exp(-(a*Xgrid**2 + b*Zgrid**2))

        if(teste_type==6):

            f       = sin(x**12*z**10)       
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgrid,Zgrid))
   
            #F[:,:] = np.sin((Xgrid**12)*(Zgrid**10))
            
        if(teste_type==7):

            f       = sin((2*pi*x)*(2*pi*z))*sin((2*pi*z))*sin((2*pi*x))       
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgrid,Zgrid))            
            
        return F
#==============================================================================

#==============================================================================
    def f_ext(self,teste):
    
        teste_type         = teste.teste_type
        Xext               = teste.Xext
        Zext               = teste.Zext
        nptx_ext           = teste.nptx_ext
        nptz_ext           = teste.nptz_ext
        x,z                = symbols('x z', real=True)
        Xgridext, Zgridext = np.meshgrid(Xext,Zext)
        F                  = np.zeros((nptx_ext,nptz_ext))
        
        if(teste_type==1):
            
            f       = sin(pi*x)*sin(pi*z)
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgridext,Zgridext))
            
            #F[:,:] = np.sin(np.pi*Xgridext)*np.sin(np.pi*Zgridext)
        
        if(teste_type==2):
            
            f       = x**4 + z**4 
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgridext,Zgridext))

            #F[:,:] = Xgridext**4 + Zgridext**4

        if(teste_type==3):

            factor  = 1000   
            f       = sin(factor*x)*sin(factor*z)        
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgridext,Zgridext))

            #factor = 1000   
            #F[:,:] = np.sin(factor*Xgridext)*np.sin(factor*Zgridext)
            
        if(teste_type==4):

            f       = x**2 + z**2
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgridext,Zgridext))

            #F[:,:] = Xgridext**2 + Zgridext**2

        if(teste_type==5):
            
            a       = 1000
            b       = 3000
            f       = exp(-(a*x**2+b*z**2)) 
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgridext,Zgridext))

            #a      = 1000
            #b      = 3000
            #F[:,:] = np.exp(-(a*Xgridext**2 + b*Zgridext**2))

        if(teste_type==6):

            f       = sin(x**12*z**10)       
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgridext,Zgridext))
            
            #F[:,:] = np.sin((Xgridext**12)*(Zgridext**10))
            
        if(teste_type==7):

            f       = sin((2*pi*x)*(2*pi*z))*sin((2*pi*z))*sin((2*pi*x))       
            Fsymbol = lambdify((x, z), f, 'numpy')
            F[:,:]  = np.transpose(Fsymbol(Xgridext,Zgridext))

        return F
#==============================================================================

#==============================================================================
    def df2(self,teste):
        
        teste_type   = teste.teste_type
        X            = teste.X
        Z            = teste.Z
        Xgrid, Zgrid = np.meshgrid(X,Z)
        x,z          = symbols('x z', real=True)
        nptx         = teste.nptx
        nptz         = teste.nptz 
        DF2exact     = np.zeros((nptx,nptz))
        
        if(teste_type==1):
            
            f              = sin(pi*x)*sin(pi*z)
            lap            = diff(diff(f,x),x)+ diff(diff(f,z),z)
            DF2exactsymbol = lambdify((x, z), lap, 'numpy')
            DF2exact[:,:]  = np.transpose(DF2exactsymbol(Xgrid,Zgrid))
          
            #DF2exact = -2*(np.pi**2)*np.sin(np.pi*Xgrid)*np.sin(np.pi*Zgrid)
                    
        if(teste_type==2):
    
            f              = x**4 + z**4
            lap            = diff(diff(f,x),x)+ diff(diff(f,z),z)
            DF2exactsymbol = lambdify((x, z), lap, 'numpy')
            DF2exact[:,:]  = np.transpose(DF2exactsymbol(Xgrid,Zgrid))
            
            #DF2exact = 12*Xgrid**2 + 12*Zgrid**2
            
        if(teste_type==3):
            
            factor         = 1000   
            f              = sin(factor*x)*sin(factor*z)
            lap            = diff(diff(f,x),x)+ diff(diff(f,z),z)
            DF2exactsymbol = lambdify((x, z), lap, 'numpy')
            DF2exact[:,:]  = np.transpose(DF2exactsymbol(Xgrid,Zgrid))
        
            #factor   = 1000   
            #DF2exact = -2*(factor**2)*np.sin(factor*Xgrid)*np.sin(factor*Zgrid)
    
        if(teste_type==4):

            f              = x**2 + z**2
            lap            = diff(diff(f,x),x)+ diff(diff(f,z),z)
            DF2exactsymbol = lambdify((x, z), lap, 'numpy')
            DF2exact[:,:]  = np.transpose(DF2exactsymbol(Xgrid,Zgrid))
                
            #DF2exact = 2*Xgrid**0 + 2*Zgrid**0

        if(teste_type==5):

            a              = 1000
            b              = 3000            
            f              = exp(-(a*x**2+b*z**2))
            lap            = diff(diff(f,x),x)+ diff(diff(f,z),z)
            DF2exactsymbol = lambdify((x, z), lap, 'numpy')
            DF2exact[:,:]  = np.transpose(DF2exactsymbol(Xgrid,Zgrid))

            #a = 1000
            #b = 3000
            #DF2exact = np.exp(-(a*Xgrid**2 + b*Zgrid**2))*(4*a*a*Xgrid**2 - 2*a - 2*b + 4*b*b*Zgrid**2)

        if(teste_type==6):

            f              = sin(x**12*z**10)
            lap            = diff(diff(f,x),x)+ diff(diff(f,z),z)
            DF2exactsymbol = lambdify((x, z), lap, 'numpy')
            DF2exact[:,:]  = np.transpose(DF2exactsymbol(Xgrid,Zgrid))

            #DF2exact = 6*Xgrid**10*Zgrid**8*(15*Xgrid**2 + 22*Zgrid**2)*np.cos(Xgrid**12*Zgrid**10) -4*Xgrid**22*Zgrid**18*(25*Xgrid**2 + 36*Zgrid**2)*np.sin(Xgrid**12*Zgrid**10)

        if(teste_type==7):

            f              = sin((2*pi*x)*(2*pi*z))*sin((2*pi*z))*sin((2*pi*x))       
            lap            = diff(diff(f,x),x)+ diff(diff(f,z),z)
            DF2exactsymbol = lambdify((x, z), lap, 'numpy')
            DF2exact[:,:]  = np.transpose(DF2exactsymbol(Xgrid,Zgrid))
            
        return DF2exact
#==============================================================================