#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
import math
import scipy.special
import sys
from   numpy import linalg as la
from scipy.linalg        import lu_factor, lu_solve
from scipy.linalg        import solve
from sympy               import Matrix
#==============================================================================

#==============================================================================
# Print Configurations
#==============================================================================
np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def assemble_system(Mvalue,neq,numbercoef,hvalue,point_list):
#==============================================================================

#==============================================================================
# Montagem do Sistema
#==============================================================================
    Asis         = np.zeros((neq,numbercoef))
    bsis         = np.zeros((neq,1))
    
    list_eq_type = []
        
    list_axes   = point_list[0]
    naxes       = len(list_axes)
    mvalue      = naxes
    
    list_diag   = point_list[1]
    ndiag       = len(list_diag)
    
    list_others = point_list[2]
    nothers     = len(list_others)
        
    conteq = 0
    contl1 = 0
#==============================================================================

#==============================================================================
# For d=1 até M
#==============================================================================       
    for dcont in range(0,Mvalue):
                
        d      = dcont + 1
        conteq = conteq + 1
        contc1 = 0
                
        vec1 = np.array([2,4,4])
        vec2 = np.array([4,4])

        if(dcont==0): bsis[dcont,0] = 1.0
        
        list_eq_type.append(0)
        
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

        for i in range(1,contd):
            
            list_eq_type.append(1)
            
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
#==============================================================================

#==============================================================================        
    return Asis,bsis,list_eq_type
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def assemble_system_precond(Mvalue,neq,numbercoef,hvalue,point_list):
#==============================================================================

#==============================================================================
# Montagem do Sistema
#==============================================================================
    Asis         = np.zeros((neq,numbercoef))
    bsis         = np.zeros((neq,1))
    
    list_eq_type = []

    list_axes   = point_list[0]
    naxes       = len(list_axes)
    mvalue      = naxes
    
    list_diag   = point_list[1]
    ndiag       = len(list_diag)
    
    list_others = point_list[2]
    nothers     = len(list_others)
        
    conteq = 0
    contl1 = 0
#==============================================================================

#==============================================================================
# For d=1 até M
#==============================================================================       
    for dcont in range(0,Mvalue):
                
        d      = dcont + 1
        conteq = conteq + 1
        contc1 = 0
                        
        vec1     = np.array([1,2,2])
        com_fact = 2
    
        if(dcont==0): bsis[dcont,0] = 1
        
        list_eq_type.append(0)
        
        for m1 in range(0,naxes):
            
            m1loc = list_axes[m1][0]
            a1    = (m1loc)**(2*d)
            a2    = 1.0
            a3    = vec1[0]
                        
            Asis[contl1,contc1] = a3*(a1/a2)
            contc1              = contc1 + 1 
                
        for m1 in range(0,ndiag):
            
            m1loc = list_diag[m1][0]
            a1    = (m1loc)**(2*d)
            a2    = 1
            a3    = vec1[1]
     
            Asis[contl1,contc1] = a3*(a1/a2)
            contc1              = contc1 + 1

        for m1 in range(0,nothers):
    
            m1loc = list_others[m1][0]
            m2loc = list_others[m1][1]
            a1    = (m1loc)**(2*d) + (m2loc)**(2*d)
            a2    = 1.0
            a3    = vec1[2]
      
            Asis[contl1,contc1] = a3*(a1/a2)
            contc1              = contc1 + 1
#==============================================================================

#==============================================================================
# For i=1 até int(d/2)
#==============================================================================
        contd  = int(d/2) + 1
        contl1 = contl1 + 1

        for i in range(1,contd):
            
            list_eq_type.append(1)

            contc1 = mvalue

            for m1 in range(0,ndiag):
                
                m1loc = list_diag[m1][0]
                a1    = (m1loc)**(2*d)
                a2    = 1.0
                a3    = 1.0
                a4    = 1.0
             
                Asis[contl1,contc1] = a4*a3*(a1/a2) 
                contc1              = contc1 + 1        

            for m1 in range(0,nothers):
                
                m1loc = list_others[m1][0]
                m2loc = list_others[m1][1]
                a1    = (m1loc)**(2*(d-i))*(m2loc)**(2*i) 
                a2    = (m2loc)**(2*(d-i))*(m1loc)**(2*i)
                a3    = 1.0
                a4    = 1.0
                a5    = 1.0
                                
                Asis[contl1,contc1] = a5*a4*((a1+a2)/a3) 
                contc1              = contc1 + 1

            conteq = conteq + 1
            
            contl1 = contl1 + 1
#==============================================================================

#==============================================================================        
    return Asis,bsis,list_eq_type
#==============================================================================

#==============================================================================
# Resolvendo Sistema de Minimização
#==============================================================================
def sol_system(Asismod,bsismod,loc_strategy,exttrap):
#==============================================================================

#==============================================================================
    ncond = la.cond(Asismod,2) 
    
    print('')
    print('====================================================')
    print('The Condition Number of System Matrix is: %.3e'%ncond)
    print('====================================================')
    print('')
    
    if(Asismod.shape[0]==Asismod.shape[1]):
    
        det   = la.det(Asismod)
        print('')
        print('====================================================')
        print('The Determinant of System Matrix is: %.3e'%det)
        print('====================================================')
        print('')
#==============================================================================

#==============================================================================
    nlin    = Asismod.shape[0]
    ncol    = Asismod.shape[1]
    csissol = []
    vtype   = []
        
    if(nlin==ncol):

        try: 

            csis1 = la.solve(Asismod,bsismod)            
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
       
        try: 

            csis3, rcsis, rankcsis, singcsis = la.lstsq(Asismod,bsismod,rcond=None)             
            vtype.append(0)

        except:
            
            csis3 = 10**(10)*np.ones((Asismod.shape[0],1))
            vtype.append(0)
        
        
        if(exttrap==0):
        
            csissol  = [csis1,csis2]
            sisttype = ['np.linalg.solve','lu_solve']
    
        else:
        
            csissol  = [csis1,csis2,csis3]
            sisttype = ['np.linalg.solve','lu_solve','la.lstsq']
            
    else:

        csis1, rcsis, rankcsis, singcsis = la.lstsq(Asismod,bsismod,rcond=None)        
        csissol   = [csis1]
        vtype     = [0]
        sisttype  = ['la.lstsq']
        
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
# Remove Linhas de Zeros
#==============================================================================
def remove_zeros(Asis,bsis,neqsys):
#==============================================================================
    zero_list = []
        
    for i in range(0,neqsys):
            
        locsum = Asis[i,:].sum()
            
        if(locsum==0):
                
            zero_list.append(i)
         
    Asis = np.delete(Asis,zero_list,0)
    bsis = np.delete(bsis,zero_list,0)
    
    return Asis, bsis
#==============================================================================

#==============================================================================
# Remove Linhas de Zeros
#==============================================================================
def linear_anlysys(Asis,numbercoefsym,list_pair_sym):
#==============================================================================
    
    R, ind_col        = Matrix(Asis).rref()
    list_ind_col      = list(ind_col)
    nlist_ind_col     = len(list_ind_col)
    list_not_ind_col  = list(np.arange(0,numbercoefsym))
    list_ind_col_coef = []
    
    for i in range(0,nlist_ind_col):
    
        pos     = int(list_ind_col[i])
        element = list_pair_sym[pos]
        list_ind_col_coef.append(element)

        if(pos in list_not_ind_col): list_not_ind_col.remove(pos)

    list_not_ind_col_coef = []
    nlist_not_ind_col     = len(list_not_ind_col)

    for i in range(0,nlist_not_ind_col):
    
        pos     = int(list_not_ind_col[i])
        element = list_pair_sym[pos]
        list_not_ind_col_coef.append(element)

    print('')
    print('====================================================')
    print('List of Linearly Independent Coefficients: ')
    print(list_ind_col_coef)
    print('====================================================')
    print('')

    nlist_ind_col_coef = len(list_ind_col_coef)
    print('')
    print('====================================================')
    print('Number of Linearly Independent Coefficients: %d '%nlist_ind_col_coef)
    print('====================================================')
    print('')

    print('')
    print('====================================================')
    print('List of Linearly Dependent Coefficients: ')
    print(list_not_ind_col_coef)
    print('====================================================')
    print('')

    nlist_not_ind_col_coef= len(list_not_ind_col_coef)
    print('')
    print('====================================================')
    print('Number of Linearly Dependent Coefficients: %d '%nlist_not_ind_col_coef)
    print('====================================================')
    print('')

    return
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(mvalue,exttrap,hvalue):
#==============================================================================
    allow_print = 1
    order       = 2*mvalue
    exttrapmax  = int(0.5*(mvalue**2+mvalue))
    pre_cond    = 1
    
    if(exttrap>exttrapmax):
        
        sys.exit('Add too much points than stencil support!')

    base_list   = []
    select_list = []
    
    for i in range(1,mvalue+1):
        
        a    = i
        b    = 0
        pair = (a,b)
        base_list.append(pair)
    
    for i in range(1,mvalue+1):
        
        for j in range(i,mvalue+1):
        
            a    = i
            b    = j
            pair = (a,b)
            base_list.append(pair)
    
    totalpoints = mvalue + exttrap
    
    for i in range(0,totalpoints):
        
        select_list.append(base_list[i])
        
    list_axes   = []
    list_diag   = []
    list_others = []

    numbercoefsym = len(select_list)
    
    list_pair_sym = select_list.copy()

    for i in range(0,numbercoefsym):
           
        a1 = list_pair_sym[i][0]
        b1 = list_pair_sym[i][1]
           
        if(i<mvalue): list_axes.append(list_pair_sym[i])
           
        if(a1==b1 and i>=mvalue): list_diag.append(list_pair_sym[i])
           
        elif(a1!=b1 and i>=mvalue):
               
            list_others.append(list_pair_sym[i])
        
    point_list = [list_axes,list_diag,list_others]
    
    list_pair_sym = []
    
    for m1 in range(0,len(list_axes)):
        
        list_pair_sym.append(list_axes[m1])
        
    for m1 in range(0,len(list_diag)):
        
        list_pair_sym.append(list_diag[m1])
    
    for m1 in range(0,len(list_others)):
        
        list_pair_sym.append(list_others[m1])
    
    list_pair  = list()
    
    for i in range(1,mvalue+1):
        
        a    = i
        b    = 0
        pair = (a,b)
        list_pair.append(pair)

    for i in range(1,mvalue+1):
        
        for j in range(1,mvalue+1):
            
            a    = i
            b    = j
            pair = (a,b)
            list_pair.append(pair)
            
    real_extra   = len(list_diag) + len(list_others)
    cross_points = mvalue
    
    if(mvalue%2==0):
        
        neq = mvalue + (mvalue**2)/4
    
    else:

        neq = mvalue + (mvalue**2-1)/4

    neq = int(neq)
    
    for m1 in range(mvalue,100):
            
        if(m1%2==0):
        
            neqnew = m1 + (m1**2)/4
    
        else:

            neqnew = m1 + (m1**2-1)/4
                
        if(pre_cond==0): Asisteste,bsis, list_eq_type = assemble_system(int(m1),int(neqnew),numbercoefsym,hvalue,point_list)
        if(pre_cond==1): Asisteste,bsis, list_eq_type = assemble_system_precond(int(m1),int(neqnew),numbercoefsym,hvalue,point_list)
        
        R, ind_col        = Matrix(Asisteste).rref()
        nlist_ind_col     = len(list(ind_col))

        if(nlist_ind_col==Asisteste.shape[1]):
            
            neqnew = int(neqnew)
            mnew   = int(m1)
            break
        
    if(neq<numbercoefsym):
        
        mvaluesys = mnew
        neqsys    = neqnew

    else:
        
        mvaluesys = mvalue
        neqsys    = neq
        
    numbercoef    = len(list_pair)
    mstencil      = np.zeros((2*mvalue+1,2*mvalue+1))
    vstencil      = np.zeros(numbercoef)
    
    print('')
    print('====================================================================================================')
    print('Cross Points  : %d'%cross_points)
    print('Extra Points: : %d'%real_extra)
    print('====================================================================================================')
    print('')
    
    print('')
    print('====================================================================================================')
    print('Possible Coefficient(s) to be calculated is(are):')
    print(base_list)
    print('====================================================================================================')
    print('')
    
    print('')
    print('====================================================================================================')
    print('The Coefficient(s) that will be calculated is(are):')
    print(list_pair_sym)
    print('====================================================================================================')
    print('')
            
    print('')
    print('===================================================')
    print('Order              :   %d'%order)    
    print('Number of Equations:   %d'%neqsys)
    print('Number of Variables:   %d'%numbercoefsym)
    print('===================================================')
    print('')
        
    if(pre_cond==0): Asis,bsis,list_eq_type = assemble_system(mvaluesys,neqsys,numbercoefsym,hvalue,point_list)
    if(pre_cond==1): Asis,bsis,list_eq_type = assemble_system_precond(mvaluesys,neqsys,numbercoefsym,hvalue,point_list)
    
    #linear_anlysys(Asis,numbercoefsym,list_pair_sym)
    
    loc_strategy = 1
   
    if(loc_strategy==0):
        
        Asis, bsis = remove_zeros(Asis,bsis,neqsys)
        if(pre_cond==0): csis = sol_system(Asis,bsis,loc_strategy,exttrap)
        if(pre_cond==1): csis = (1/hvalue**2)*sol_system(Asis,bsis,loc_strategy,exttrap)
    
    if(loc_strategy==1):
    
        Ateste  = np.zeros((Asis.shape[1],Asis.shape[1]))
        bteste  = np.zeros((Asis.shape[1],1))
        contcol = 0
    
        for m1 in range(0,Asis.shape[0]):
           
            if(contcol==Asis.shape[1]):
            
                break
        
            if(list_eq_type[m1]==0):
            
                Ateste[contcol,:] = Asis[m1,:]
                bteste[contcol,0] = bsis[m1,0]
                contcol           = contcol + 1
      
        for m1 in range(0,Asis.shape[0]):
            
           if(contcol==Asis.shape[1]):
            
               break
        
           if(list_eq_type[m1]==1):
            
               Ateste[contcol,:] = Asis[m1,:]
               bteste[contcol,0] = bsis[m1,0]
               contcol           = contcol + 1
        
        R, ind_col        = Matrix(Ateste).rref()
        list_ind_col      = list(ind_col)
        nlist_ind_col     = len(list_ind_col)
        if(pre_cond==0):  csis = sol_system(Ateste,bteste,loc_strategy,exttrap)
        if(pre_cond==1):  csis = (1/hvalue**2)*sol_system(Ateste,bteste,loc_strategy,exttrap)        
    
    if(loc_strategy==2):
    
        Ateste  = np.zeros((mvaluesys,Asis.shape[1]))
        bteste  = np.zeros((mvaluesys,1))
        contcol = 0
    
        for m1 in range(0,Asis.shape[0]):
            
            if(contcol==mvaluesys):
            
                break
        
            if(list_eq_type[m1]==0):

                Ateste[contcol,:] = Asis[m1,:]
                bteste[contcol,0] = bsis[m1,0]
                contcol           = contcol + 1
                
        if(pre_cond==0):  csis = sol_system(Ateste,bteste,loc_strategy,exttrap)
        if(pre_cond==1):  csis = (1/hvalue**2)*sol_system(Ateste,bteste,loc_strategy,exttrap)
        
    for m1 in range(0,numbercoefsym):
        
        a1 = list_pair_sym[m1][0]
        b1 = list_pair_sym[m1][1]
        
        for m2 in range(0,numbercoef):
            
            a2 = list_pair[m2][0]
            b2 = list_pair[m2][1]
           
            if(a1==a2 and b1==b2):
                
                vstencil[m2] = csis[m1]
            
            if(a1==b2 and b1==a2):

                vstencil[m2] = csis[m1]     
#==============================================================================

#==============================================================================    
    if(allow_print==1):
        
        print('')
        print(Asis)
        
        print('')
        print(bsis)
        
        print('')
        print(csis)
#==============================================================================

#==============================================================================
# Calcula Termo c00
#==============================================================================
    c00 = 0

    for i in range(0,numbercoef):
        
        c00 = c00 - 4*vstencil[i] 
#==============================================================================
    
#==============================================================================
# Distibui Pesos pelo Stencil
#==============================================================================
    nmeio                 = int(0.5*(2*mvalue+1))
    mstencil[nmeio,nmeio] = c00
    
    for j in range(0,mvalue+1):
        
        for i in range(0,mvalue):
            
            mstencil[nmeio+1+i,nmeio]   = vstencil[i]
            mstencil[nmeio-1-i,nmeio]   = vstencil[i]
            mstencil[nmeio-j,nmeio+1+i] = vstencil[j*mvalue + i]
            mstencil[nmeio+j,nmeio+1+i] = vstencil[j*mvalue + i]
            mstencil[nmeio-j,nmeio-1-i] = vstencil[j*mvalue + i]
            mstencil[nmeio+j,nmeio-1-i] = vstencil[j*mvalue + i]
#==============================================================================
    
#==============================================================================
    return mstencil
#==============================================================================