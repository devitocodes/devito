#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as np
import sys
#==============================================================================

#==============================================================================
# Funcoes Auxiliares
#==============================================================================

#==============================================================================
def decomplu(a,n):
            
    p = np.zeros(n)
        
    for i in range(0,n):
        
        p[i] = i
     
    for k in range(0,n):
        
        amax = 0.0
        im   = int(p[k])
         
        for i in range(k,n):
             
            ii = int(p[i])
            
            for j in range(0,k):
                
               a[ii,k] = a[ii,k] - a[ii,j]*a[int(p[j]),k]
    
            if(np.abs(a[ii,k])>amax):
                
               amax = np.abs(a[ii,k])
               im   = ii
               ik   = i
        
        p[ik] = int(p[k])
        p[k]  = int(im)
        
        for j in range(k+1,n):
            
            ik = int(p[k])
            
            for i in range(0,k):
                
               a[ik,j] = a[ik,j]-a[ik,i]*a[int(p[i]),j]
            
            a[int(p[j]),k] = a[int(p[j]),k]/a[int(p[k]),k]
      
    return a,p
#==============================================================================

#==============================================================================
def solve_lu(a,b,p,n):
    
    n = int(n)
    
    x = np.zeros(n)
    
    x[0] = b[int(p[0])]
    
    det = 0.0
    
    for i in range(1,n):
        
        x[i] = b[int(p[i])]
        
        for j in range(0,i): 
        
            x[i] = x[i] - a[int(p[i]),j]*x[j]
    
    det    = a[int(p[n-1]),n-1]
    x[n-1] = x[n-1]/a[int(p[n-1]),n-1]
   
    for i in range(n-2,-1,-1):
        
        for j in range(i+1,n):
            
            x[i] = x[i] - a[int(p[i]),j]*x[j]
         
        det  = det*a[int(p[i]),i]
        x[i] = x[i]/a[int(p[i]),i]
        
    return x
#==============================================================================

#==============================================================================
def set_matrix(nd,l,n,gama):
    
    nd = int(nd) 
    l  = int(l) 
    b  = np.zeros(nd)
    a  = np.zeros((nd,nd))

    pi = np.pi
    h  = pi/n
        
    x = np.zeros(n)
    
    for i in range(0,n):

        x[i] = (i+1)*h
      
    a[0,:]    =  1
    a[0,nd-1] =  0
    b[0]      =  1 
      
    for i in range(0,l):
          
        a[i+1,0]    = 0
        a[i+1,nd-1] = 0
        b[i+1]      = gama**(2*(i+1))

        for j in range(0,nd-2): 
            
            a[i+1,j+1] = (j+1)**(2*(i+1))
                              
    mi = -1
    
    for i in range(0,n):
        
        k         = i + l + 1
        a[k,0]    = 1
        a[k,nd-1] = mi
        b[k]      = np.cos(gama*x[i])
        
        for j in range(0,nd-2):
        
            a[k,j+1] = np.cos((j+1)*x[i]) 
         
        mi = - mi
                 
        return a, b
#==============================================================================

#==============================================================================
# Main Function
#==============================================================================
def calccoef(nordem,rval,knumber):

    m    = nordem
    gama = rval
    l    = knumber
    n    = m-l+1
    nd   = m+2
        
    p  = np.zeros(nd)
    b  = np.zeros(nd)
    c  = np.zeros(nd)
    a  = np.zeros((nd,nd))

    a, b = set_matrix(nd,l,n,gama)
    a,p  = decomplu(a,nd) 
    c    = solve_lu(a,b,p,nd)
        
    ntotal = 2*m + 1
    
    vcoef = np.zeros(ntotal)
    
    nmeio = int(0.5*ntotal)
        
    vcoef[nmeio] = c[0]   
 
    for i in range(0,nmeio):
        
        vcoef[i]          = c[nmeio-i] 
        vcoef[ntotal-i-1] = c[nmeio-i]
            
    return vcoef
#==============================================================================