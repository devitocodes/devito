#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
import math
import scipy.special
import sys
from numpy               import linalg as la
from scipy.linalg        import lu_factor, lu_solve
from scipy.sparse.linalg import lsqr,cg
import pyamg
from scipy.sparse import csr_matrix
#==============================================================================

#==============================================================================
# Print Configurations
#==============================================================================
np.set_printoptions(formatter={'float': '{: 0.4e}'.format})
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def assemble_system(mvalue,nvalue,neq,numbercoef,hvalue,point_list):
#==============================================================================

#==============================================================================
# Montagem do Sistema
#==============================================================================
    Asis         = np.zeros((neq,numbercoef))
    bsis         = np.zeros((neq,1))
    
    list_axes   = point_list[0]
    naxes       = len(list_axes)
    
    list_diag   = point_list[1]
    ndiag       = len(list_diag)
    
    list_others = point_list[2]
    nothers     = len(list_others)
    
    conteq = 0
    contl1 = 0
    contn  = 0
#==============================================================================

#==============================================================================
# For d=1 até M
#==============================================================================       
    for dcont in range(0,mvalue):
                
        d      = dcont + 1
        conteq = conteq + 1
        contc1 = 0
        
        if(dcont==0): bsis[dcont,0] = 1
        
        vec1 = np.array([2,4,4])
        vec2 = np.array([4,4])
        
        for m1 in range(0,naxes):
            
            m1loc = list_axes[m1][0]
            a1    = (m1loc*hvalue)**(2*d)
            a2    = np.math.factorial(2*d)
            a3    = vec1[0]
            
            Asis[contl1,contc1] = a3*(a1/a2)
            contc1              = contc1 + 1 
                
        for m1 in range(0,ndiag):
            
            m1loc = list_diag[m1][0]
            a1    = (m1loc*hvalue)**(2*d)
            a2    = np.math.factorial(2*d)
            a3    = vec1[2]
     
            Asis[contl1,contc1] = a3*(a1/a2)
            contc1              = contc1 + 1

        for m1 in range(0,nothers):
    
            m1loc = list_others[m1][0]
            m2loc = list_others[m1][1]
            a1    = (m1loc*hvalue)**(2*d) + (m2loc*hvalue)**(2*d)
            a2    = np.math.factorial(2*d)
            a3    = vec1[1]
      
            Asis[contl1,contc1] = a3*(a1/a2)
            contc1              = contc1 + 1
#==============================================================================

#==============================================================================
# For i=1 até int(d/2)
#==============================================================================
        contd  = int(d/2) + 1
        contl1 = contl1 + 1

        if(contn<=nvalue-1):

            for i in range(1,contd):
                
                contc1 = mvalue

                for m1 in range(0,ndiag):
                
                    m1loc = list_diag[m1][0]
                    a1    = (m1loc*hvalue)**(2*d)
                    a2    = np.math.factorial(2*d)
                    a3    = scipy.special.binom(2*d, 2*i)
                    a4    = vec2[1] 
             
                    Asis[contl1,contc1] = a4*a3*(a1/a2) 
                    contc1              = contc1 + 1        

                for m1 in range(0,nothers):
                
                    m1loc = list_others[m1][0]
                    m2loc = list_others[m1][1]
                    a1    = (m1loc*hvalue)**(2*(d-i))*(m2loc*hvalue)**(2*i) 
                    a2    = (m2loc*hvalue)**(2*(d-i))*(m1loc*hvalue)**(2*i)
                    a3    = np.math.factorial(2*d)
                    a4    = scipy.special.binom(2*d, 2*i)
                    a5    = vec2[0]
                
                    Asis[contl1,contc1] = a5*a4*((a1+a2)/a3) 
                    contc1              = contc1 + 1

                conteq = conteq + 1
            
                contl1 = contl1 + 1
        
        contn  = contn + 1
#==============================================================================

#==============================================================================    
    return Asis,bsis
#==============================================================================

#==============================================================================
# Resolvendo Sistema de Minimização
#==============================================================================
def sol_system(Asismod,bsismod):
#==============================================================================

#==============================================================================
    ncond = la.cond(Asismod,2) 
    
    print('')
    print('====================================================')
    print('The Condition Number of System Matrix is: %.3e'%ncond)
    print('====================================================')
    print('')
    
    det   = la.det(Asismod)
    print('')
    print('====================================================')
    print('The Determinant of System Matrix is: %.3e'%det)
    print('====================================================')
    print('')
#==============================================================================

#==============================================================================
    vtype = []

    try: 

        csis1 = np.linalg.solve(Asismod,bsismod)
        vtype.append(0)
            
    except:
            
        csis1 = 10**(10)*np.ones((Asismod.shape[0],1))
        vtype.append(0)

    try: 

        csis2 = lu_solve((lu_factor(Asismod)), bsismod)
        vtype.append(0)
    
    except:
            
        csis2 = 10**(10)*np.ones((Asismod.shape[0],1))
        vtype.append(0)
    
    csissol   = [csis1,csis2]    
    sisttype  = ['np.linalg.solve','lu_solve']
    nsisttype = 0
    nsis      = len(csissol)
    csisnorm  = 10**(100000)
        
    for k1 in range(0,nsis):
        
        if(np.isfinite(csissol[k1].sum())==True):            
            
            locnorm = la.norm(csissol[k1],2)
                    
            if(locnorm<=csisnorm):
            
                nsisttype = k1 
                csisnorm  = locnorm
                
                if(vtype[k1]==0): csis = csissol[k1][:,0]
                if(vtype[k1]==1): csis = csissol[k1][:]

    print('')
    print('====================================')
    print('System Solver: %s'%sisttype[nsisttype])
    print('====================================')
    print('')
        
    return csis
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(mvalue,nvalue,hvalue):
#==============================================================================
    allow_print = 0

    order = 2*mvalue
    
    if(nvalue%2==0):
            
        neq = mvalue + (nvalue**2)/4
        
    else:

        neq = mvalue + (nvalue**2-1)/4
        
    neq        = int(neq)
    list_pair  = list()
    list_pair2 = list()

    for i in range(1,mvalue+1):
        
        a    = i
        b    = 0
        pair = (a,b)
        list_pair.append(pair)
        
        a    = 0
        b    = i
        pair = (a,b)
        list_pair2.append(pair)

    for i in range(1,nvalue):
        
        for j in range(1,nvalue-i+1):
            
            a    = i
            b    = j
            pair = (a,b)
            list_pair.append(pair)
            
            a    = -i
            b    = j
            pair = (a,b)
            list_pair2.append(pair)

    numbercoef    = len(list_pair)
    mstencil      = np.zeros((2*mvalue+1,2*mvalue+1))
    vstencil      = np.zeros(numbercoef)
    list_pair_sym = list_pair.copy()
    list_cont     = []
    
    for m1 in range(0,numbercoef):
        
        a1 = list_pair[m1][0]
        b1 = list_pair[m1][1]
                
        for m2 in range(m1+1,numbercoef):
        
            a2 = list_pair[m2][0]
            b2 = list_pair[m2][1]
            
            if(a1==b2 and b1==a2):
                
                list_cont.append(m2)
    
    
    nremove     = len(list_cont)
    cont_remove = 0
    
    for m1 in range(0,nremove):
        
        list_pair_sym.remove(list_pair_sym[list_cont[m1]-cont_remove])
        cont_remove = cont_remove + 1
        
    numbercoefsym = len(list_pair_sym)
    
    list_axes   = []
    list_diag   = []
    list_others = []
    
    for i in range(0,numbercoefsym):
        
        a1 = list_pair_sym[i][0]
        b1 = list_pair_sym[i][1]
        
        if(i<mvalue): list_axes.append(list_pair_sym[i])
        
        if(a1==b1 and i>=mvalue): list_diag.append(list_pair_sym[i])
        
        elif(a1!=b1 and i>=mvalue):
            
            list_others.append(list_pair_sym[i])
            
    point_list = [list_axes,list_diag,list_others]
    
    print('')
    print('====================================================================================================')
    print('mvalue: : %d'%mvalue)
    print('nvalue: : %d'%nvalue)
    print('====================================================================================================')
    print('')
    
    print('')
    print('====================================================================================================')
    print('The Coefficient(s) that will be calculated is(are):')
    print(list_pair)
    print('====================================================================================================')
    print('')
    
    print('')
    print('=================================================================')
    print('The Number of Coefficient(s) Without Simmetry is(are): %d'%numbercoef)
    print('=================================================================')
    print('')
    
    print('')
    print('====================================================================================================')
    print('The Coefficient(s) that will be calculated with Simmetry is(are):')
    print(list_pair_sym)
    print('====================================================================================================')
    print('')
    
    print('')
    print('=================================================================')
    print('The Number of Coefficient(s) Without Simmetry is(are): %d'%numbercoefsym)
    print('=================================================================')
    print('')
    
    print('')
    print('===================================================')
    print('Order              :   %d'%order)    
    print('Number of Equations:   %d'%neq)
    print('===================================================')
    print('')
    
    Asis,bsis = assemble_system(mvalue,nvalue,neq,numbercoefsym,hvalue,point_list)
    csis      = sol_system(Asis,bsis)
    
    for m1 in range(0,numbercoefsym):
        
        a1 = list_pair_sym[m1][0]
        b1 = list_pair_sym[m1][1]
        
        for m2 in range(0,numbercoef):
            
            a2 = list_pair[m2][0]
            b2 = list_pair[m2][1]
            
            if(a1==a2 and b1==b2):
                
                vstencil[m2] = csis[m1]
            
            if(a1==b2 and b1==a1):
            
                vstencil[m2] = csis[m1]    
#==============================================================================

#==============================================================================    
    if(allow_print==1):
        
        print('')
        print(Asis)
        
        print('')
        print(bsis)
#==============================================================================

#==============================================================================
# Calcula Termo c00
#==============================================================================
    c00   = 0
    cont1 = 0
    
    cte1 = 4
    cte2 = 8
    cte3 = 4

    for i in range(1,mvalue+1):

        c00   = c00 - cte1*vstencil[cont1]
        cont1 = cont1 + 1
    
    for i in range(1,nvalue):
         
        for j in range(1,nvalue-i+1):
            
            if(i==j): cte = cte3
            if(i!=j): cte = cte2
            

            c00   = c00 - cte2*vstencil[cont1]
            cont1 = cont1 + 1    
#==============================================================================
    
#==============================================================================
# Distibui Pesos pelo Stencil
#==============================================================================
    nmeio                 = int(0.5*(2*mvalue+1))
    mstencil[nmeio,nmeio] = c00
    
    
    for i in range(0,mvalue):
        
        a = list_pair2[i][0]
        b = list_pair2[i][1]
        mstencil[nmeio+a,nmeio+b] = vstencil[i]
        mstencil[nmeio+a,nmeio-b] = vstencil[i]
        mstencil[nmeio-a,nmeio+b] = vstencil[i]
        mstencil[nmeio-a,nmeio-b] = vstencil[i]
    
    for i in range(1,mvalue+1):
        
        mstencil[nmeio+i,nmeio] = mstencil[nmeio,nmeio+i]
        mstencil[nmeio-i,nmeio] = mstencil[nmeio,nmeio+i]
    
    for j in range(0,mvalue):
        
        for i in range(0,mvalue):
            
            mstencil[j,i] = mstencil[i,j]
        
    for j in range(0,mvalue):
   
        mstencil[:,2*mvalue-j] = mstencil[:,j]
        mstencil[2*mvalue-j,:] = mstencil[j,:]
#==============================================================================
    
#==============================================================================
    return mstencil
#==============================================================================