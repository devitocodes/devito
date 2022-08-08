#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
import math
import sys
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(mvalue,nvalue,rvalx,rvaly):

    #===========================================
    rval = min(rvalx,rvaly)
    #===========================================
        
    #===========================================
    list_pair = list()
    
    list_sis  = list()
    
    list_pair.append((0,0))

    nvar = 0
             
    for i in range(1,mvalue+1):
        
        a = i
        b = 0
        pair = (a,b)
        
        list_pair.append(pair)
        
        nvar = nvar + 1
    
    for i in range(1,nvalue):
                
        for j in range(1,nvalue-i+1):
            
            a = i
            b = j
            pair = (a,b)
            list_pair.append(pair)
            
            nvar = nvar + 1
    
    if(nvalue%2==0):
        
        neq = mvalue + (nvalue**2)/4
    
    else:

        neq = mvalue + (nvalue**2-1)/4
    
    neq = int(neq)

    list_sis = list_pair[1:]
    #===========================================
    
    #===========================================
    #print()
    #print('O numero de variaveis sao: ',nvar)
    #print('O numero de equacoes  sao: ',neq)
    #print('Estamos calculando os pesos para os seguintes pontos: ',list_sis)
    #print()
    #===========================================
    
    #===========================================
    Asis  = np.zeros((neq,nvar))
    
    bsis  = np.zeros((neq,1))
    
    csis  = np.zeros((neq,1))
    #===========================================    
    
    #===========================================
    for j in range(0,mvalue):
        
        bsis[j,0] = rval**(2*(j+1)-2)
        
        for i in range(0,mvalue):
            
            Asis[j,i] = (i+1)**(2*(j+1))
    
        py = mvalue
                
        for m1 in range(1,nvalue):

            for m2 in range(1,nvalue-m1+1):
                                
                Asis[j,py] = 2*(m1)**(2*(j+1))
    
                py = py + 1
    #===========================================
        
    #===========================================
    px = mvalue - 1
        
    for j1 in range(1,nvalue+1):
        
        pf = int((j1)/2) + 1
                
        py = mvalue
                
        for j2 in range(1,pf):
                        
            px = px + 1

            py = mvalue
            
            n1 = math.factorial(j1)
            n2 = math.factorial(2*j2)
            n3 = math.factorial(2*(j1-j2))
            n4 = rval**(2*j1-2)
            
            d1 = math.factorial(2*j1)
            d2 = math.factorial(j2)
            d3 = math.factorial(j1-j2)
            d4 = 2
            
            bsis[px,0] = (n1*n2*n3*n4)/(d1*d2*d3*d4)

            for m1 in range(1,nvalue):
                                
                for m2 in range(1,nvalue-m1+1):
                                        
                    a = (m1)**(2*((j1)-(j2)))
                    b = (m2)**(2*(j2))
                                
                    Asis[px,py] = a*b
    
                    py = py + 1
    #===========================================
        
    #===========================================    
    Asis1 = np.zeros((neq,nvar))

    Asis1 = Asis.copy()
    
    npsis = len(list_sis)
    
    npdel = list()
    
    for i in range(0,npsis):
        
        a0 = list_sis[i][0]
        b0 = list_sis[i][1]
        
        for j in range(i+1,npsis):
            
            a1 = list_sis[j][0]
            b1 = list_sis[j][1]
            
            if(a0==b1 and b0==a1):
    
                Asis1[:,i] = Asis1[:,i] + Asis1[:,j]        
                                
                npdel.append(j)
    
    nnpdel = len(npdel)
    
    nvarnew = nvar - nnpdel
            
    list_sis_new = list_sis.copy()
    
    Asis2 = np.zeros((neq,nvarnew))
     
    npdel = np.sort(npdel)    

    for i in range(0,nnpdel):
                
        Asis1 = np.delete(Asis1,[npdel[i]],1)
        
        del list_sis_new[npdel[i]]
        
        for j in range(i+1,nnpdel):
            
            npdel[j] = npdel[j] - 1
    
    Asis2 = Asis1.copy()
    #===========================================
        
    #===========================================  
    #print()
    #print('O numero de variaveis atulizadas sao: ',nvarnew)
    #print('O numero de equacoes  sao: ',neq)
    #print('Pontos do Sistema: ',list_sis_new)
    #print()
    #===========================================
        
    #===========================================
    csis = np.linalg.solve(Asis2,bsis)
    #===========================================
    
    #===========================================    
    nvartotal = len(list_pair)
    
    nlistnew  = len(list_sis_new)
    
    coefglob = np.zeros(nvartotal)
        
    coefglob[0] = 0.0
    
    for i in range(1,nvartotal):
        
        a0 = list_pair[i][0]
        b0 = list_pair[i][1]
        
        for j in range(0,nlistnew):
            
            a1 = list_sis_new[j][0]
            b1 = list_sis_new[j][1]
            
            if(a0==a1 and b0==b1):

                coefglob[i] = csis[j,0]
                
            if(a0==b1 and b0==a1):

                coefglob[i] = csis[j,0]    
    #===========================================

    #===========================================    
    for i in range(1,nvartotal):
        
        coefglob[0] = coefglob[0] + (-4*coefglob[i])
    #===========================================

    #===========================================
    sum_coef = 0
    
    for k in range(1,coefglob.shape[0]):
        
        i = list_pair[k][0]
        j = list_pair[k][1]
        
        sum_coef = sum_coef + ((-1)**(i+j-1) + 1)*coefglob[k]

    stab = 1/np.sqrt(sum_coef)

    if(rval>stab): print('Infringindo a condição de estabilidade para o método! Cuidado!')
    #===========================================
   
    #===========================================    
    dimmat = 2*mvalue + 1
    
    mcoef = np.zeros((dimmat,dimmat))
    
    mcoef[mvalue,mvalue] = coefglob[0]
    
    for i in range(1,mvalue+1):
        
        mcoef[mvalue,mvalue+i] = coefglob[i]
        mcoef[mvalue,mvalue-i] = coefglob[i]
        mcoef[mvalue+i,mvalue] = coefglob[i]
        mcoef[mvalue-i,mvalue] = coefglob[i]
    
    for i in range(mvalue+1,nvartotal):
        
        px           = list_pair[i][0] + mvalue
        py           = list_pair[i][1] + mvalue
        mcoef[px,py] = coefglob[i]

        px           = -list_pair[i][0] + mvalue
        py           = -list_pair[i][1] + mvalue
        mcoef[px,py] = coefglob[i]
    
    for i in range(0,mvalue):
        
        for j in range(0,mvalue):
                        
            mcoef[i,mvalue+1+j] = mcoef[i,mvalue-1-j]
     
            mcoef[mvalue+1+i,mvalue-1-j] = mcoef[mvalue+1+i,mvalue+1+j]     
    #===========================================

    return mcoef
#==============================================================================