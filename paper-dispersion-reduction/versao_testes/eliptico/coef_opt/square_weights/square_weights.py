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
np.set_printoptions(formatter={'float': '{: 0.1e}'.format})
#==============================================================================

#==============================================================================
# Sistema de Minimização - Pré Condicionado
#==============================================================================
def assemble_system_precond(M,neq,numbercoef,knumber,hvalue,coef_square_type):
#==============================================================================

#==============================================================================
# Montagem do Sistema
#==============================================================================
    Asis         = np.zeros((neq,numbercoef))
    bsis         = np.zeros((neq,1))
    list_eq_type = list()

    conteq = 0
    contl1 = 0
    
    for d in range(1,M+1):

        conteq = conteq + 1
        contc1 = 0
#==============================================================================
# For d=1 até M
#==============================================================================              
        if(coef_square_type==0):
            
            a0   = (2*hvalue**(2*d))/(np.math.factorial(2*d))
            vec1 = np.array([1,0,0])
            vec2 = np.array([0,0])
            if(d==1): bsis[d-1,0] = 1/a0

        if(coef_square_type==1):
            
            a01  = (2*hvalue**(2*d))
            vec1 = (1/np.math.factorial(2*d))*np.array([1,2,2])
            vec2 = (1/np.math.factorial(2*d))*np.array([2,2])
            if(d==1): bsis[d-1,0] = 1/a01
        
        #if(coef_square_type==1):
            
            #a01  = (2*hvalue**(2*d))/(np.math.factorial(2*d))
            #vec1 = np.array([1,2,2])
            #vec2 = np.array([1,1])
            #if(d==1): bsis[d-1,0] = 1/a01
        
        for m1 in range(1,knumber+1):
            
            a1 = (m1)**(2*d)
            a2 = 1.0
            a3 = vec1[0]
            
            Asis[contl1,contc1] = a3*(a1/a2)
                                      
            contc1 = contc1 + 1
       
        for m1 in range(1,knumber+1):
      
            for m2 in range(m1+1,knumber+1):
         
                a1 = (m1)**(2*d) + (m2)**(2*d)
                a2 = 1.0
                a3 = vec1[1]
        
                Asis[contl1,contc1] = a3*(a1/a2)

                contc1 = contc1 + 1

        for m1 in range(1,knumber+1):
            
            a1 = (m1)**(2*d)
            a2 = 1.0
            a3 = vec1[2]
    
            Asis[contl1,contc1] = a3*(a1/a2)

            contc1 = contc1 + 1
        
        list_eq_type.append(0)
#==============================================================================
# For i=1 até int(d/2)
#==============================================================================
        contd  = int(d/2)
        contl1 = contl1 + 1 

        for i in range(1,contd+1):
        
            contc1 = knumber
        
            for m1 in range(1,knumber+1):
      
                for m2 in range(m1+1,knumber+1):
                    
                    a1 = (m1)**(2*(d-i))*(m2)**(2*i) 
                    a2 = (m2)**(2*(d-i))*(m1)**(2*i)
                    a3 = 1.0
                    a4 = 1.0
                    a5 = vec2[0]
                    Asis[contl1,contc1] = a5*a4*((a1+a2)/a3) 

                    contc1 = contc1 + 1

            for m1 in range(1,knumber+1):
                
                a1 = (m1)**(2*d)
                a2 = 1.0
                a3 = 1.0
                a4 = vec2[1] 
               
                Asis[contl1,contc1] = a4*a3*(a1/a2) 

                contc1 = contc1 + 1
        
            conteq = conteq + 1
            
            contl1 = contl1 + 1
            
            list_eq_type.append(1)
#==============================================================================

#==============================================================================    
    return Asis,bsis,list_eq_type
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def assemble_system(M,neq,numbercoef,knumber,hvalue,coef_square_type):
#==============================================================================

#==============================================================================
# Montagem do Sistema
#==============================================================================
    Asis         = np.zeros((neq,numbercoef))
    bsis         = np.zeros((neq,1))
    list_eq_type = list()

    conteq = 0
    contl1 = 0
    
    for d in range(1,M+1):

        conteq = conteq + 1
        contc1 = 0
#==============================================================================
# For d=1 até M
#==============================================================================       
        if(d==1): bsis[d-1,0] = 1
        
        if(coef_square_type==0):
        
            vec1 = np.array([2,0,0])
            vec2 = np.array([0,0])
        
        if(coef_square_type==1):
        
            vec1 = np.array([2,4,4])
            vec2 = np.array([4,4])
        
        for m1 in range(1,knumber+1):
            
            a1 = (m1*hvalue)**(2*d)
            a2 = np.math.factorial(2*d)
            a3 = vec1[0]
            
            Asis[contl1,contc1] = a3*(a1/a2)
                                      
            contc1 = contc1 + 1
       
        for m1 in range(1,knumber+1):
      
            for m2 in range(m1+1,knumber+1):
         
                a1 = (m1*hvalue)**(2*d) + (m2*hvalue)**(2*d)
                a2 = np.math.factorial(2*d)
                a3 = vec1[1]
        
                Asis[contl1,contc1] = a3*(a1/a2)

                contc1 = contc1 + 1

        for m1 in range(1,knumber+1):
            
            a1 = (m1*hvalue)**(2*d)
            a2 = np.math.factorial(2*d)
            a3 = vec1[2]
    
            Asis[contl1,contc1] = a3*(a1/a2)

            contc1 = contc1 + 1
        
        list_eq_type.append(0)
#==============================================================================
# For i=1 até int(d/2)
#==============================================================================
        contd  = int(d/2)
        contl1 = contl1 + 1 

        for i in range(1,contd+1):
      
            contc1 = knumber
        
            for m1 in range(1,knumber+1):
      
                for m2 in range(m1+1,knumber+1):
                    
                    a1 = (m1*hvalue)**(2*(d-i))*(m2*hvalue)**(2*i) 
                    a2 = (m2*hvalue)**(2*(d-i))*(m1*hvalue)**(2*i)
                    a3 = np.math.factorial(2*d)
                    a4 = scipy.special.binom(2*d, 2*i)
                    a5 = vec2[0]
                    Asis[contl1,contc1] = a5*a4*((a1+a2)/a3) 

                    contc1 = contc1 + 1

            for m1 in range(1,knumber+1):
                
                a1 = (m1*hvalue)**(2*d)
                a2 = np.math.factorial(2*d)
                a3 = scipy.special.binom(2*d, 2*i)
                a4 = vec2[1] 
               
                Asis[contl1,contc1] = a4*a3*(a1/a2) 

                contc1 = contc1 + 1
        
            conteq = conteq + 1
            
            contl1 = contl1 + 1
            list_eq_type.append(1)
#==============================================================================

#==============================================================================    
    return Asis,bsis,list_eq_type
#==============================================================================

#==============================================================================

#==============================================================================
# Resolvendo Sistema de Minimização
#==============================================================================
def sol_system(Asismod,bsismod,strategy_loc,precondsys):
#==============================================================================

#==============================================================================
    ncond = la.cond(Asismod,2) 
    
    print('')
    print('====================================================')
    if(precondsys==0): print('The System Matrix is not preconditioned!')
    if(precondsys==1): print('The System Matrix is preconditioned!')
    print('====================================================')
    print('')
    
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

        try: 

            csis3, rcsis, rankcsis, singcsis = la.lstsq(Asismod,bsismod,rcond=None) 
            vtype.append(0)
            
        except:
            
            csis3 = 10**(10)*np.ones((Asismod.shape[0],1))
            vtype.append(0)

        try: 

            ml    = pyamg.ruge_stuben_solver(csr_matrix(Asismod))
            csis4 = ml.solve(bsismod, tol=1e-10)
            vtype.append(1)
            
            #ml        = pyamg.smoothed_aggregation_solver(csr_matrix(Asismod), bsismod, max_coarse=10)
            #residuals = []
            #x0        = np.random.rand(Asismod.shape[0])
            #csis4     = ml.solve(b=bsismod, x0=x0, tol=1e-10, residuals=residuals)
            #csis4     = ml.solve(b=bsismod, x0=x0, tol=1e-10, residuals=residuals, accel='cg')
            #vtype.append(1)

        except:
     
            csis4 = 10**(10)*np.ones((Asismod.shape[0],1))
            vtype.append(0)
            
        csissol  = [csis1,csis2,csis3,csis4]
        
        if(strategy_loc==-1): csissol  = [csis1]

        sisttype = ['np.linalg.solve','lu_solve','la.lstsq','pyamg']
    
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
# Sistema de Minimização
#==============================================================================
def calccoef(knumber,hvalue,square_strategy,coef_square_type):
#==============================================================================

#==============================================================================    
    allow_print = 1
    precondsys  = 0

    numbercoef = int(0.5*knumber*(knumber+3))
    mstencil   = np.zeros((2*knumber+1,2*knumber+1))
    vstencil   = np.zeros(numbercoef)

    print('')
    print('====================================================')
    print('The Number of Coefficient(s) is(are): %d'%numbercoef)
    print('====================================================')
    print('')
    
    list_pair      = list()
    list_pair2     = list()
    list_pair_type = list()
    contcol        = 0

    for i in range(1,knumber+1):
        
        a    = i
        b    = 0
        pair = (a,b)
        list_pair.append(pair)
        
        a    = 0
        b    = i
        pair = (a,b)
        list_pair2.append(pair)

        a       = i
        b       = 0
        pair    = (a,b,0,contcol)
        list_pair_type.append(pair)
        contcol = contcol + 1

    for i in range(1,knumber+1):
        
        for j in range(i+1,knumber+1):
            
            a    = i
            b    = j
            pair = (a,b)
            list_pair.append(pair)
 
            a    = -i
            b    = j
            pair = (a,b)
            list_pair2.append(pair)

            a       = i
            b       = j
            pair    = (a,b,1,contcol)
            list_pair_type.append(pair)
            contcol = contcol + 1
        
    for i in range(1,knumber+1):
        
        a    = i
        b    = i
        pair = (a,b)
        list_pair.append(pair)
        
        a    = -i
        b    = i
        pair = (a,b)
        list_pair2.append(pair)
       
        a       = i
        b       = i
        pair    = (a,b,1,contcol)
        list_pair_type.append(pair)
        contcol = contcol + 1
    
    list_pair_type = sorted(list_pair_type)

    print('')
    print('====================================================================================================')
    print('The Coefficient(s) that will be calculated is(are):')
    print(list_pair)
    print('====================================================================================================')
    print('')
    
    minimalorder = 0
    minimalneq   = 0
    maximalorder = 0
    maximalneq   = 0
    
    for m in range(1,100):
        
        if((m%2)==0):
            
            neq = 0.25*(m**2+4*m)
            
        else:
                
            neq = 0.25*(m**2+4*m-1)
        
        if(neq>numbercoef):
      
            minimalorder = minimalorder_last
            minimalneq   = minimalneq_last
            maximalorder = 2*m
            maximalneq   = neq
             
            break
        
        else:
            
            minimalorder_last = 2*m
            minimalneq_last   = neq
    
    print('')
    print('===================================================')
    print('Minimal Order              :   %d'%minimalorder)    
    print('Minimal Number of Equations:   %d'%minimalneq)
    print('===================================================')
    print('')

    print('')
    print('===================================================')
    print('Maximal Order              :   %d'%maximalorder)    
    print('Maximal Number of Equations:   %d'%maximalneq)
    print('===================================================')
    print('')
    
    M_max   = int(maximalorder/2)
    neq_max = int(maximalneq)
    dif_eq  = neq_max - numbercoef
 
    M_min   = int(minimalorder/2)
    neq_min = int(minimalneq)
    dif_var = numbercoef - neq_min
    
    M_new   = int(numbercoef)
    neq_new = 0
    
    if((M_new%2)==0):
        
        neq_new = 0.25*(M_new**2+4*M_new)
        
    else:
            
        neq_new = 0.25*(M_new**2+4*M_new-1)
    
    neq_new     = int(neq_new) 
    dif_eq_new  = neq_new - numbercoef
    new_order   = int(2*M_new) 
    
    print('')
    print('===================================================')
    print('New Order              :       %d'%new_order)    
    print('New Number of Equations:       %d'%neq_new)
    print('===================================================')
    print('')
    
    M_classic = knumber
      
    if((M_classic%2)==0):
        
        neq_clasic = 0.25*(M_classic**2+4*M_classic)
        
    else:
            
        neq_clasic = 0.25*(M_classic**2+4*M_classic-1)
    
    neq_clasic    = int(neq_clasic) 
    classic_order = int(2*M_classic) 
    
    print('')
    print('===================================================')
    print('Classical Order              : %d'%classic_order)    
    print('Classical Number of Equations: %d'%neq_clasic)
    print('===================================================')
    print('')
    
    if(coef_square_type==0):
    
        strategy_loc = -1
            
        a=-1
        print('')
        print('===================================================')
        print('Strategy: Classical Scheme')
        print('===================================================')
        print('')
        
        print('')
        print('=================================================================================================')
        print('Solve a Square System with %d Equation(s) and %d Variable(s) with Approximation Order %d.'%(knumber,knumber,classic_order))
        print('=================================================================================================')
        print('')
    
    if(square_strategy==0 and coef_square_type==1):
    
        if(dif_var==0): 
        
            strategy_loc = 1
            
            a=-1
            print('')
            print('===================================================')
            print('Strategy: Maintain Equation(s) and Variable(s)')
            print('===================================================')
            print('')
            
            print('')
            print('=================================================================================================')
            print('Solve a Square System with %d Equation(s) and %d Variable(s) with Approximation Order %d.'%(minimalneq,numbercoef,minimalorder))
            print('=================================================================================================')

            print('')
            
        else:
    
            strategy_loc = 2
            
            a=-1
            print('')
            print('===================================================')
            print('Strategy: Remove Coefficient(s)')
            print('===================================================')
            print('')
            
            print('')
            print('=================================================================================================')
            print('Eliminate %d Variable(s) and Solve a Square System with %d Equation(s) and %d Variable(s) with Approximation Order %d'%(dif_var,minimalneq,numbercoef-dif_var,minimalorder))
            print('=================================================================================================')
            print('')
    
    if(square_strategy==1 and coef_square_type==1):
    
        if(dif_eq==0): 

            strategy_loc = 3
            
            a=-1
            print('')
            print('===================================================')
            print('Strategy: Maintain Equation(s) and Variable(s)')
            print('===================================================')
            print('')
            
            print('')
            print('=================================================================================================')
            print('Solve a Square System with %d Equation(s) and %d Variable(s) with Approximation Order %d.'%(maximalneq,numbercoef,maximalorder))
            print('=================================================================================================')
            print('')
            
        elif(dif_eq>0):    
    
            strategy_loc = 4
                              
            a=-1
            print('')
            print('===================================================')
            print('Strategy: Remove Equation(s)')
            print('===================================================')
            print('')
            
            print('')
            print('=================================================================================================')
            print('Eliminate %d Equation(s) and Solve a Square System with %d Equation(s) and %d Variable(s) with Approximation Order %d'%(dif_eq,maximalneq-dif_eq,numbercoef,maximalorder))
            print('=================================================================================================')
            print('')
     
    if(square_strategy==2 and coef_square_type==1):
        
        strategy_loc = 5

        a=-1
        print('')
        print('===================================================')
        print('Strategy: Maintain Equation(s) and Variable(s)')
        print('===================================================')
        print('')
  
        print('')
        print('=================================================================================================')
        print('Solve a Non-Square System with %d Equation(s) and %d Variable(s) with Maximum Approximation Order %d'%(maximalneq,numbercoef,maximalorder))
        print('=================================================================================================')
        print('')

    if(square_strategy==3 and coef_square_type==1):
        
        strategy_loc = 6

        a=-1
        print('')
        print('===================================================')
        print('Strategy: Remove Equation(s)')
        print('===================================================')
        print('')
     
        print('')
        print('=================================================================================================')
        print('Eliminate %d Equation(s) and Solve a Square System with %d Equation(s) and %d Variable(s) with Approximation Order %d'%(dif_eq_new,neq_new-dif_eq_new,numbercoef,new_order))
        print('=================================================================================================')
        print('')              
#==============================================================================

#==============================================================================
# Strategy -1
#==============================================================================
    if(strategy_loc==-1):
        
        M                      = M_classic
        neq                    = neq_clasic
        if(precondsys==0): Asis,bsis,list_eq_type = assemble_system(M,neq,numbercoef,knumber,hvalue,coef_square_type)    
        if(precondsys==1): Asis,bsis,list_eq_type = assemble_system_precond(M,neq,numbercoef,knumber,hvalue,coef_square_type)    

        Asismod = np.zeros((knumber,knumber))
        bsismod = np.zeros((knumber,1))

        atypecol = []
        
        nlin   = Asis.shape[0]
        ncol   = Asis.shape[1]
        
        Asismodtemp = np.zeros((knumber,Asis.shape[1]))
        contlin     = 0
        contcol     = 0 
        
        for k1 in range(0,nlin):
            
            sel_lin = Asis[k1,:]
            
            sumlin = sel_lin.sum()
            
            if(sumlin!=0):
                
                Asismodtemp[contlin,:] = Asis[k1,:]
                bsismod[contlin,0]     = bsis[k1,0]
                contlin                = contlin + 1
        
        for k1 in range(0,ncol):

            sel_col = Asismodtemp[:,k1]
            
            sumcol = sel_col.sum()
            
            if(sumcol!=0):

                Asismod[:,contcol] = Asismodtemp[:,k1]
                contcol            = contcol + 1

        csis                      = sol_system(Asismod,bsismod,strategy_loc,precondsys)
        vstencil[0:csis.shape[0]] = csis[:]
#==============================================================================

#==============================================================================
# Strategy 0 - Strategy Loc 1
#==============================================================================
    if(strategy_loc==1):
        
        M                      = M_min
        neq                    = neq_min
        if(precondsys==0): Asis,bsis,list_eq_type = assemble_system(M,neq,numbercoef,knumber,hvalue,coef_square_type)    
        if(precondsys==1): Asis,bsis,list_eq_type = assemble_system_precond(M,neq,numbercoef,knumber,hvalue,coef_square_type)    

        Asismod = Asis
        bsismod = bsis

        csis                      = sol_system(Asismod,bsismod,strategy_loc,precondsys)
        vstencil[0:csis.shape[0]] = csis[:]
#==============================================================================

#==============================================================================
# Strategy 0 - Strategy Loc 2
#==============================================================================
    if(strategy_loc==2):
       
       M                      = M_min
       neq                    = neq_min
       if(precondsys==0): Asis,bsis,list_eq_type = assemble_system(M,neq,numbercoef,knumber,hvalue,coef_square_type)    
       if(precondsys==1): Asis,bsis,list_eq_type = assemble_system_precond(M,neq,numbercoef,knumber,hvalue,coef_square_type)    

       Asismod = np.zeros((neq_min,numbercoef-dif_var))
       bsismod = np.zeros((neq_min,1))
       numvar  = len(list_pair_type)
       contdif = 0
       contcol = numbercoef-dif_var 

       solv_order = list()
       
       bsismod[:,0] = bsis[:,0]
       
       for m1 in range(numvar-1,-1,-1):
           
           pair = list_pair_type[m1]
           
           typevar   = pair[2]
           colnumber = pair[3] 
                      
           if(typevar==0 or contdif>=dif_var): 
               
               Asismod[:,contcol-1] = Asis[:,colnumber]
               solv_order.append((pair[0],pair[1],contcol-1))
              
               contcol = contcol - 1
               
           if(typevar==1 and contdif<=dif_var):
               
               contdif = contdif + 1
    
       solv_order = sorted(solv_order)
       
       csis = sol_system(Asismod,bsismod,strategy_loc,precondsys)

       npairs = len(list_pair)
       ncoef  = len(solv_order)
      
       for i in range(0,npairs):
           
           a1 = int(list_pair[i][0])
           b1 = int(list_pair[i][1])
           
           for j in range(0,ncoef):
               
               a2 = int(solv_order[j][0])
               b2 = int(solv_order[j][1])
               
               if(a1==a2 and b1==b2):
                   
                   vstencil[i] = csis[j]
#==============================================================================

#==============================================================================
# Strategy 1 - Strategy Loc 3
#==============================================================================
    if(strategy_loc==3):
        
        M                      = M_max
        neq                    = neq_max
        if(precondsys==0): Asis,bsis,list_eq_type = assemble_system(M,neq,numbercoef,knumber,hvalue,coef_square_type)    
        if(precondsys==1): Asis,bsis,list_eq_type = assemble_system_precond(M,neq,numbercoef,knumber,hvalue,coef_square_type)    

        Asismod = Asis
        bsismod = bsis

        csis                      = sol_system(Asismod,bsismod,strategy_loc,precondsys)
        vstencil[0:csis.shape[0]] = csis[:]
#==============================================================================

#==============================================================================
# Strategy 1 - Strategy Loc 4
#==============================================================================
    if(strategy_loc==4):

        M                      = M_max
        neq                    = neq_max
        if(precondsys==0): Asis,bsis,list_eq_type = assemble_system(M,neq,numbercoef,knumber,hvalue,coef_square_type)    
        if(precondsys==1): Asis,bsis,list_eq_type = assemble_system_precond(M,neq,numbercoef,knumber,hvalue,coef_square_type)    

        Asismod  = np.zeros((neq_max-dif_eq,numbercoef))
        bsismod  = np.zeros((numbercoef,1))
        numeq    = len(list_eq_type)
        contdif  = 0
        contline = neq_max-dif_eq
                
        for m1 in range(numeq-1,-1,-1):
            
            eqtype = list_eq_type[m1]
            
            if(eqtype==0 or contdif>=dif_eq): 
                
                Asismod[contline-1,:] = Asis[m1,:]
                bsismod[contline-1,:] = bsis[m1,:]

                contline = contline - 1
                
            if(eqtype==1 and contdif<=dif_eq):
                
                contdif = contdif + 1
        
        csis                      = sol_system(Asismod,bsismod,strategy_loc,precondsys)
        vstencil[0:csis.shape[0]] = csis[:]
#==============================================================================

#==============================================================================
# Strategy 2
#==============================================================================
    if(strategy_loc==5):
        
        M                      = M_max
        neq                    = neq_max
        if(precondsys==0): Asis,bsis,list_eq_type = assemble_system(M,neq,numbercoef,knumber,hvalue,coef_square_type)    
        if(precondsys==1): Asis,bsis,list_eq_type = assemble_system_precond(M,neq,numbercoef,knumber,hvalue,coef_square_type)    

        Asismod = Asis
        bsismod = bsis
        
        csis                      = sol_system(Asismod,bsismod,strategy_loc,precondsys)
        vstencil[0:csis.shape[0]] = csis[:]
#==============================================================================

#==============================================================================
# Strategy 3
#==============================================================================
    if(strategy_loc==6):
        
        M                      = M_new
        neq                    = neq_new
        if(precondsys==0): Asis,bsis,list_eq_type = assemble_system(M,neq,numbercoef,knumber,hvalue,coef_square_type)    
        if(precondsys==1): Asis,bsis,list_eq_type = assemble_system_precond(M,neq,numbercoef,knumber,hvalue,coef_square_type)    

        Asismod  = np.zeros((neq_new-dif_eq_new,numbercoef))
        bsismod  = np.zeros((numbercoef,1))
        
        numeq    = len(list_eq_type)
        contdif  = 0
        contline = neq_new-dif_eq_new
                
        for m1 in range(numeq-1,-1,-1):
            
            eqtype = list_eq_type[m1]
            
            if(eqtype==0 or contdif>=dif_eq_new): 
                
                Asismod[contline-1,:] = Asis[m1,:]
                bsismod[contline-1,:] = bsis[m1,:]

                contline = contline - 1
                
            if(eqtype==1 and contdif<=dif_eq):
                
                contdif = contdif + 1
        
        csis                      = sol_system(Asismod,bsismod,strategy_loc,precondsys)
        vstencil[0:csis.shape[0]] = csis[:]
#==============================================================================

#==============================================================================    
    if(allow_print==1):
        
        print('')
        print(Asis)
        
        #print('')
        #print(bsis)
        
        print('')
        print(csis)
        
        #print('')
        #print(Asismod)
        
        #print('')
        #print(bsismod)
#==============================================================================

#==============================================================================
# Calcula Termo c00
#==============================================================================
    c00   = 0
    cont1 = 0
    
    cte1 = 4
    cte2 = 8
    cte3 = 4

    for i in range(1,knumber+1):

        c00   = c00 - cte1*vstencil[cont1]
        cont1 = cont1 + 1
         
    for i in range(1,knumber+1):
         
        for j in range(i+1,knumber+1):

            c00   = c00 - cte2*vstencil[cont1]
            cont1 = cont1 + 1
    
    for i in range(1,knumber+1):

        c00   = c00 - cte3*vstencil[cont1]
        cont1 = cont1 + 1
#==============================================================================
    
#==============================================================================
# Distibui Pesos pelo Stencil
#==============================================================================
    nmeio                 = int(0.5*(2*knumber+1))
    mstencil[nmeio,nmeio] = c00
        
    for i in range(0,numbercoef):
        
        a = list_pair2[i][0]
        b = list_pair2[i][1]
        mstencil[nmeio+a,nmeio+b] = vstencil[i]
        mstencil[nmeio+a,nmeio-b] = vstencil[i]
        mstencil[nmeio-a,nmeio+b] = vstencil[i]
        mstencil[nmeio-a,nmeio-b] = vstencil[i]
    
    for i in range(1,knumber+1):
        
        mstencil[nmeio+i,nmeio] = mstencil[nmeio,nmeio+i]
        mstencil[nmeio-i,nmeio] = mstencil[nmeio,nmeio+i]
    
    for j in range(0,knumber):
        
        for i in range(0,knumber):
            
            mstencil[j,i] = mstencil[i,j]
        
    for j in range(0,knumber):
   
        mstencil[:,2*knumber-j] = mstencil[:,j]
        mstencil[2*knumber-j,:] = mstencil[j,:]
#==============================================================================
    
#==============================================================================
    return mstencil
#==============================================================================