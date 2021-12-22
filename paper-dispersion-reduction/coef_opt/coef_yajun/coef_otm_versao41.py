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
def exchange(x,n,sgn,xmax,sgnmax):
    
    n = int(n)
      
    if(xmax<x[0]):
        
        if(sgn*sgnmax<0):
            
            x[0] = xmax
          
        else:
            
            x[0:n-1] = x[1:n]
            x[n-1] = xmax
         
    if(xmax>x[n-1]):
         
        a = ((-1)**(n-1))*sgn*sgnmax 
        
        if(a>0):
            
            x[n-1] = xmax
        
        else:
        
            x[1:n] = x[0:n-1]
            x[0]   = xmax
    
    i1 = 0
    i2 = n-1
        
    while(True):
        
        i3 = i2 - i1

        k = int((i1+i2)/2)

        if(i3==1):
        
            break
   
        if(xmax<x[k]):
        
            i2 = k
    
        else:
        
            i1 = k
        
    a = ((-1)**(i1-1))*sgn*sgnmax
        
    if(a>0):
        
         x[i1] = xmax
    
    else:
        
         x[i2] = xmax

    return x
#==============================================================================

#==============================================================================
def g(x,c,n):
    
    n = int(n)
    
    g = 0.0
    
    g = c[0]
    
    for k in range(1,n):
        
        g = g + c[k]*np.cos(k*x)
     
    return g
#==============================================================================

#==============================================================================
def error(x,gama,c,m):
    
    m = int(m)
    
    error = np.cos(gama*x)- g(x,c,m+1)
    
    return error
#==============================================================================

#==============================================================================
def max_fib(a,b,gama,coef,m):
    
    m = int(m)
    
    r = (np.sqrt(5)+1)/2
    
    eps = 10**(-5)
    
    while(True):
    
        c = b - (b-a)/r
    
        d = a + (b-a)/r     
    
        if(np.abs(c-d)<eps):
        
            break
   
        a1 = error(c,gama,coef,m)
    
        a2 = error(d,gama,coef,m)
    
        if(a1>a2):
        
            b = d
    
        else:
   
            a = c
    
    max_fib = (a+b)/2
    
    return max_fib
#==============================================================================

#==============================================================================
def min_fib(a,b,gama,coef,m):
      
    m = int(m)
    
    r = (np.sqrt(5)+1)/2
    
    eps = 10**(-5)
    
    while(True):
    
        c = b - (b-a)/r
    
        d = a + (b-a)/r

        if(np.abs(c-d)<eps):
        
            break
    
        a1 = error(c,gama,coef,m) 
    
        a2 = error(d,gama,coef,m)
    
        if(a1<a2):

            b = d

        else:
    
            a = c
    
    min_fib = (a+b)/2
    
    return min_fib
#==============================================================================

#==============================================================================
def set_matrix(x,nd,l,n,gama,ini):
    
    nd = int(nd) 
    l  = int(l) 
    b  = np.zeros(nd)
    a  = np.zeros((nd,nd))
    
    if(ini==0):
    
        pi = np.pi
        h  = pi/n
        
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
     
    if(ini==0):
        
        return a, b, x
    
    else:
    
        return a, b
#==============================================================================

#==============================================================================
def glob_max(gama,c,m):
    
    k    = 100
    pi   = np.pi
    h    = pi/k
    f1   = error(0,gama,c,m)
    f2   = error(h,gama,c,m)
    fm   = f1
    xmax = 0
            
    for i in range(1,k):
        
        a  = (i-1)*h
        b  = (i+1)*h
        f3 = error(b,gama,c,m)
        
        a1 = f2 - f1
        a2 = f3 - f2
        a3 = a1*a2
        
        if(a3<=0):
        
            if(a1>0):
                
                xm = max_fib(a,b,gama,c,m)
            
            elif(a2<0):
            
                xm = max_fib(a,b,gama,c,m)
            
            else:
                
                xm = min_fib(a,b,gama,c,m)
            
            f = error(xm,gama,c,m)
            
            a4 = np.abs(f)
            a5 = np.abs(fm)
            
            if(a4>a5):
            
                fm   = f
                xmax = xm
                     
        f1 = f2
        f2 = f3
    
    a6 = np.abs(f3)
    a7 = np.abs(fm)
    
    if(a6>a7):
        
        fm   = f3
        xmax = pi
      
    sgnmax = fm/np.abs(fm)
          
    return xmax, sgnmax
#==============================================================================

#==============================================================================
# Best App
#==============================================================================
def calccoef(nordem,rval,knumber):

    m    = nordem
    gama = rval
    l    = knumber
    n    = m-l+1
    nd   = m+2
    
    eps = 10**(-5)
    h   = np.pi/200
    
    x  = np.zeros(n)
    p  = np.zeros(nd)
    b  = np.zeros(nd)
    c  = np.zeros(nd)
    a  = np.zeros((nd,nd))

    a, b, x = set_matrix(x,nd,l,n,gama,0)
    a,p     = decomplu(a,nd) 
    c       = solve_lu(a,b,p,nd)
    
    while(True):
    
        fmax         = np.abs(c[nd-1])
        xmax, sgnmax = glob_max(gama,c,m)
        erroloc      = np.abs(error(xmax,gama,c,m))-fmax
   
        if(erroloc<eps): 
        
            break
    
        sgn  = fmax/c[nd-1]
        x    = exchange(x,n,sgn,xmax,sgnmax)
        a, b = set_matrix(x,nd,l,n,gama,1)
        a, p = decomplu(a,nd)
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