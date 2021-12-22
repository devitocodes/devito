#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
import math
import sys
import time  as tm
from   scipy                   import integrate
from   scipy.integrate         import nquad
from   scipy.integrate         import dblquad
#==============================================================================

#==============================================================================
# Sistema de Minimização
#==============================================================================
def calccoef(mvalue,nvalue,rvalx,rvaly):

    #===========================================
    rval    = min(rvalx,rvaly)
    tetaint = 2*np.pi
    bint    = 2.74
    epsabs  = 10**(-5)
    epsrel  = 10**(-5)
    limit   = 20
    #===========================================
        
    #===========================================
    m0 = mvalue + int(nvalue**2/4)
        
    nhalf = int(nvalue/2) + 1
        
    list_pair = list()
    
    list_sis  = list()
    
    list_pair.append((0,0))

    nvartotal = 1
    
    for i in range(1,mvalue+1):
        
        a = i
        b = 0
        pair = (a,b)
        
        list_pair.append(pair)
        
        nvartotal = nvartotal + 1
    
    for i in range(1,nvalue):
                
        for j in range(1,nvalue-i+1):
            
            a = i
            b = j
            pair = (a,b)
                        
            list_pair.append(pair)
            
            nvartotal = nvartotal + 1
    
    nvarsis = 0
    
    minfo = list()
    
    for i in range(1,mvalue+1):
        
         a = i
         b = 0
         pair = (a,b)
        
         list_sis.append(pair)
        
         nvarsis = nvarsis + 1
    
         mvect = (nvarsis,i,0)
    
         minfo.append(mvect)
    
    for m1 in range(1,nhalf):
    
        for m2 in range(m1,nvalue-m1+1):
            
            m = mvalue + (m1-1)*nvalue - m1*(m1-1) + m2
            
            pair = (m1,m2)
        
            list_sis.append(pair)
        
            nvarsis = nvarsis + 1
   
            mvect = (nvarsis,m1,m2)
        
            minfo.append(mvect)
    
    neq = nvarsis
    #===========================================
        
    #===========================================
    #print()
    #print('O numero total de variaveis sao: ',nvartotal)
    #print('O numero de variaveis do sistema sao: ',nvarsis)
    #print('O numero de equacoes  sao: ',neq)
    #print('Pontos do Stencil: ',list_pair)
    #print('Estamos calculando os pesos para os seguintes pontos: ',list_sis)
    #print()
    #===========================================

    #===========================================
    Asis  = np.zeros((neq,nvarsis))
    
    bsis  = np.zeros((neq,1))
    
    csis  = np.zeros((neq,1))
    #===========================================    
          
    #===========================================    
    def funsistem11(beta,theta,m,rval):
      
        div = (rval**(-2))*(1-np.cos(rval*beta))
        
        a = 2
        b = - np.cos(m*beta*np.cos(theta))
        c = - np.cos(m*beta*np.sin(theta))
            
        num = a + b + c 
        
        cvalue = num/div
        
        return cvalue
    #===========================================    
    
    #===========================================    
    def funsistem12(beta,theta,m1,m2,rval):
        
        div = (rval**(-2))*(1-np.cos(rval*beta))
                
        a = 2
        b = - np.cos(m1*beta*np.cos(theta)+m2*beta*np.sin(theta))
        c = - np.cos(m1*beta*np.cos(theta)-m2*beta*np.sin(theta))
            
        num = a + b + c
        
        cvalue = num/div
        
        return cvalue
    #===========================================    
   
    #===========================================    
    def funsistem13(beta,theta,m1,m2,rval):

        div = (rval**(-2))*(1-np.cos(rval*beta))
        
        a = 4
        b = - np.cos(m1*beta*np.cos(theta)+m2*beta*np.sin(theta))
        c = - np.cos(m1*beta*np.cos(theta)-m2*beta*np.sin(theta))
        d = - np.cos(m2*beta*np.cos(theta)+m1*beta*np.sin(theta))
        e = - np.cos(m2*beta*np.cos(theta)-m1*beta*np.sin(theta))
            
        num = a + b + c + d + e
        
        cvalue = num/div
        
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem211(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem11(beta,theta,m0,rval)
        b = funsistem11(beta,theta,m1,rval)
        
        cvalue = a*b
        
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem212(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem11(beta,theta,m0,rval)
        b = funsistem12(beta,theta,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

   #===========================================    
    def funsistem213(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem11(beta,theta,m0,rval)
        b = funsistem13(beta,theta,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem221(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem12(beta,theta,m01,m02,rval)
        b = funsistem11(beta,theta,m1,rval)
        
        cvalue = a*b
        
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem222(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem12(beta,theta,m01,m02,rval)
        b = funsistem12(beta,theta,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

   #===========================================    
    def funsistem223(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem12(beta,theta,m01,m02,rval)
        b = funsistem13(beta,theta,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem231(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem13(beta,theta,m01,m02,rval)
        b = funsistem11(beta,theta,m1,rval)
        
        cvalue = a*b
        
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem232(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem13(beta,theta,m01,m02,rval)
        b = funsistem12(beta,theta,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem233(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem13(beta,theta,m01,m02,rval)
        b = funsistem13(beta,theta,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    
    
    #===========================================    
    start = tm.time()
        
    for i in range(0,nvarsis):
    
        m0  = minfo[i][0]
        m01 = minfo[i][1]
        m02 = minfo[i][2]      

        if(m0<=mvalue):
            
            res, err  = nquad(funsistem11,[[0,bint],[0,tetaint]],args=(m0,rval,))  
            
        elif(m0>mvalue and m01==m02):
        
            res, err  = nquad(funsistem12,[[0,bint],[0,tetaint]],args=(m01,m02,rval,))  
            
        else:

            res, err  = nquad(funsistem13,[[0,bint],[0,tetaint]],args=(m01,m02,rval,))  
            
        bsis[i,0] = res
    
        for j in range(0,nvarsis):
   
            m1  = minfo[j][0]
            m11 = minfo[j][1]
            m12 = minfo[j][2]
            
            if(m0<=mvalue):
            
                if(m1<=mvalue):
            
                    res, err  = nquad(funsistem211,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))

                elif(m1>mvalue and m11==m12):
                    
                    res, err  = nquad(funsistem212,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))  
                    
                else:

                    res, err  = nquad(funsistem213,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))  

            elif(m0>mvalue and m01==m02):
                
                if(m1<=mvalue):
            
                    res, err  = nquad(funsistem221,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))  

                elif(m1>mvalue and m11==m12):
                    
                    res, err  = nquad(funsistem222,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))  

                else:

                    res, err  = nquad(funsistem223,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))  

            else:
                
                if(m1<=mvalue):
                    
                    res, err  = nquad(funsistem231,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))  

                elif(m1>mvalue and m11==m12):
                    
                    res, err  = nquad(funsistem232,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))  

                else:

                    res, err  = nquad(funsistem233,[[0,bint],[0,tetaint]],args=(m0,m01,m02,m1,m11,m12,rval,))  

            Asis[i,j] = res
    
    end   = tm.time()
    print("Tempo de Geração do Sistema = %s" % (end - start))
    print('')
    #===========================================    
    
    #===========================================
    #start = tm.time()
    csis = np.linalg.solve(Asis,bsis)
    #end   = tm.time() 
    #print("Tempo de Resolução do Sistema = %s" % (end - start))
    #print('')
    #===========================================
        
    #===========================================
    coefglob = np.zeros(nvartotal)
        
    coefglob[0] = 0.0
    
    for i in range(1,nvartotal):
        
         a0 = list_pair[i][0]
         b0 = list_pair[i][1]
        
         for j in range(0,nvarsis):
            
             a1 = list_sis[j][0]
             b1 = list_sis[j][1]
            
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