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
    optsdic={}
    #optsdic={'limit': limit, 'epsrel': epsrel,'epsabs': epsabs}
    #optsdic={'limit': limit, 'epsrel': epsrel}
    #optsdic={'limit': limit, 'epsabs': epsabs}
    #optsdic={'epsrel': epsrel,'epsabs': epsabs}
    #optsdic={'limit': limit}
    #optsdic={'epsabs': epsabs}
    #optsdic={'epsrel': epsrel}
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
    def funsistem11(beta,theta,m,m1,m2,rval):
      
        div = (rval**(-2))*(1-np.cos(rval*beta))
        
        a = 2
        b = - np.cos(m*beta*np.cos(theta))
        c = - np.cos(m*beta*np.sin(theta))
            
        num = a + b + c 
        
        cvalue = num/div
        
        return cvalue
    #===========================================    
    
    #===========================================    
    def funsistem12(beta,theta,m,m1,m2,rval):
        
        div = (rval**(-2))*(1-np.cos(rval*beta))
                
        a = 2
        b = - np.cos(m1*beta*np.cos(theta)+m2*beta*np.sin(theta))
        c = - np.cos(m1*beta*np.cos(theta)-m2*beta*np.sin(theta))
            
        num = a + b + c
        
        cvalue = num/div
        
        return cvalue
    #===========================================    
   
    #===========================================    
    def funsistem13(beta,theta,m,m1,m2,rval):

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
      
        a = funsistem11(beta,theta,m0,m01,m02,rval)
        b = funsistem11(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
        
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem212(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem11(beta,theta,m0,m01,m02,rval)
        b = funsistem12(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

   #===========================================    
    def funsistem213(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem11(beta,theta,m0,m01,m02,rval)
        b = funsistem13(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem221(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem12(beta,theta,m0,m01,m02,rval)
        b = funsistem11(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
        
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem222(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem12(beta,theta,m0,m01,m02,rval)
        b = funsistem12(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

   #===========================================    
    def funsistem223(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem12(beta,theta,m0,m01,m02,rval)
        b = funsistem13(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem231(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem13(beta,theta,m0,m01,m02,rval)
        b = funsistem11(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
        
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem232(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem13(beta,theta,m0,m01,m02,rval)
        b = funsistem12(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

    #===========================================    
    def funsistem233(beta,theta,m0,m01,m02,m1,m11,m12,rval):
      
        a = funsistem13(beta,theta,m0,m01,m02,rval)
        b = funsistem13(beta,theta,m1,m11,m12,rval)
        
        cvalue = a*b
            
        return cvalue
    #===========================================    

    #===========================================        
    ainfo = np.zeros((nvarsis,nvarsis))
    binfo = np.zeros((nvarsis,1))
    
    #start = tm.time()
    
    for i in range(0,nvarsis):
    
        m0  = minfo[i][0]
        m01 = minfo[i][1]
        m02 = minfo[i][2]      

        if(m0<=mvalue):
        
            binfo[i,0] = 11
            
        elif(m0>mvalue and m01==m02):

            binfo[i,0] = 12

        else:
   
            binfo[i,0] = 13

        for j in range(0,nvarsis):
   
            m1  = minfo[j][0]
            m11 = minfo[j][1]
            m12 = minfo[j][2]
                         
            if(m0<=mvalue):
            
                if(m1<=mvalue):
            
                    ainfo[i,j] = 211

                elif(m1>mvalue and m11==m12):
                    
                    ainfo[i,j] = 212

                else:

                    ainfo[i,j] = 213

            elif(m0>mvalue and m01==m02):
                
                if(m1<=mvalue):
                    
                    ainfo[i,j] = 221


                elif(m1>mvalue and m11==m12):
                    
                    ainfo[i,j] = 222
    

                else:

                    ainfo[i,j] = 223

            else:
                
                if(m1<=mvalue):
                    
                    ainfo[i,j] = 231
    

                elif(m1>mvalue and m11==m12):
                    
                    ainfo[i,j] = 232

                else:
                    
                    ainfo[i,j] = 233
    #end   = tm.time()
    #print("Tempo de Alocação Dados = %s" % (end - start))
    #===========================================    

    #==========================================================================
    def pontopeso1(npi):
    
        pontos = np.zeros(npi)
    
        pesos = np.zeros(npi)
    
        if(npi==1):
    
            pontos[0] = 0.0
            pesos[0]  = 2.0

        if(npi==2):
     
            pontos[0] = -0.577350269189626
            pontos[1] = -pontos[0]
        
            pesos[0]  =  1.0
            pesos[1]  =  pesos[0]
    
        if(npi==3):
     
            pontos[0]  = -0.774596669241483
            pontos[1]  =  0.0
            pontos[2]  = -pontos[0]
    
            pesos[0]   = 0.555555555555556
            pesos[1]   = 0.888888888888889
            pesos[2]   = pesos[0]
        
        if(npi==4):
      
            pontos[0]  = -0.861136311594053
            pontos[1]  = -0.339981043584856
            pontos[2]  = -pontos[1]
            pontos[3]  = -pontos[0]
    
            pesos[0]   = 0.347854845137454
            pesos[1]   = 0.652145154862546
            pesos[2]   = pesos[1]
            pesos[3]   = pesos[0]
 
        if(npi==5):
    
            pontos[0]  = -0.906179845938664
            pontos[1]  = -0.538469310105683
            pontos[2]  =  0.0
            pontos[3]  = -pontos[1]
            pontos[4]  = -pontos[0]
    
            pesos[0]   = 0.236926885056189
            pesos[1]   = 0.478628670499366
            pesos[2]   = 0.568888888888889
            pesos[3]   = pesos[1]
            pesos[4]   = pesos[0]
        
        if(npi==6):
    
            pontos[0]  = -0.9324695142
            pontos[1]  = -0.6612093864
            pontos[2]  = -0.2386191860
            pontos[3]  = -pontos[2]
            pontos[4]  = -pontos[1]    
            pontos[5]  = -pontos[0]    
    
            pesos[0]   = 0.1713244923
            pesos[1]   = 0.3607615730
            pesos[2]   = 0.4679139345
            pesos[3]   = pesos[2]
            pesos[4]   = pesos[1]
            pesos[5]   = pesos[0]
    
        if(npi==7):
    
            pontos[0]  = -0.9491079123
            pontos[1]  = -0.7415311855
            pontos[2]  = -0.4058451513
            pontos[3]  =  0.0
            pontos[4]  = -pontos[2]
            pontos[5]  = -pontos[1]
            pontos[6]  = -pontos[0]
        
            pesos[0]   = 0.1294849661
            pesos[1]   = 0.2797053914
            pesos[2]   = 0.3818300505
            pesos[3]   = 0.4179591836
            pesos[4]   = pesos[2]
            pesos[5]   = pesos[1]
            pesos[6]   = pesos[0]

        if(npi==8):
    
            pontos[0]  = -0.9602898564
            pontos[1]  = -0.7966664774
            pontos[2]  = -0.5255324099
            pontos[3]  = -0.1834346424
            pontos[4]  = -pontos[3]
            pontos[5]  = -pontos[2]
            pontos[6]  = -pontos[1]
            pontos[7]  = -pontos[0]
          
            pesos[0]   = 0.1012285362
            pesos[1]   = 0.2223810344
            pesos[2]   = 0.3137066458
            pesos[3]   = 0.3626837833
            pesos[4]   = pesos[3]
            pesos[5]   = pesos[2]
            pesos[6]   = pesos[1]
            pesos[7]   = pesos[0]

        return pontos,pesos
    #==========================================================================

    #==========================================================================
    def pontopeso2(npi):
        
        pontos = np.zeros(npi)
    
        pesos = np.zeros(npi)
    
        m = int((npi+1)*0.5)

        for i in range(0,m):
        
            cte1 = np.pi*(i+1-0.25)/(npi+0.5)
        
            x = np.cos(cte1)
                    
            while(True):
        
                p1 = 1
                p2 = 0
                
                for j in range(0,npi):
                
                    p3 = p2
                    p2 = p1
                    p1 = ((2*j+1)*x*p2-j*p3)/(j+1)
            
                pp = npi*(x*p1-p2)/(x**2-1)
            
                x1 = x
                x = x1 - (p1/pp)
                
                if(abs(x-x1)<1e-15):
            
                    break
        
            pontos[i]       = -x
            pontos[npi-1-i] = -pontos[i]
        
            pesos[i]         = 2/((1-x**2)*pp**2)
            pesos[npi-1-i]   = pesos[i]
                        
        return pontos,pesos
        #======================================================================
        
    #==========================================================================
    def int2dtype1(x1,x2,y1,y2,m,m1,m2,rval,pontosx,pontosy,pesosx,pesosy,f):

        S    = 0.0
        npix = pontosx.shape[0]
        npiy = pontosy.shape[0]

        for i in range(0,npix):
        
            beta = (x2-x1)*0.5*pontosx[i]+(x2+x1)*0.5

            for j in range(0,npiy):

                theta = (y2-y1)*0.5*pontosy[j]+(y2+y1)*0.5;

                S = S + pesosx[i]*pesosy[j]*f(beta,theta,m,m1,m2,rval);

        int2dvalue = (x2-x1)*0.5*(y2-y1)*0.5*S
        
        return int2dvalue
    #==========================================================================

    #==========================================================================
    def int2dtype2(x1,x2,y1,y2,m,m1,m2,t,t1,t2,rval,pontosx,pontosy,pesosx,pesosy,f):

        S    = 0.0
        npix = pontosx.shape[0]
        npiy = pontosy.shape[0]

        for i in range(0,npix):
        
            beta = (x2-x1)*0.5*pontosx[i]+(x2+x1)*0.5

            for j in range(0,npiy):

                theta = (y2-y1)*0.5*pontosy[j]+(y2+y1)*0.5;

                S = S + pesosx[i]*pesosy[j]*f(beta,theta,m,m1,m2,t,t1,t2,rval);

        int2dvalue = (x2-x1)*0.5*(y2-y1)*0.5*S
        
        return int2dvalue
    #==========================================================================
    
    #===========================================    
    start = tm.time()
    
    npix = 15
    npiy = 15
    
    pontosx,pesosx = pontopeso2(npix)
    pontosy,pesosy = pontopeso2(npiy)
    x1   = 0
    x2   = bint
    y1   = 0
    y2   = tetaint
     
    for i in range(0,nvarsis):
    
        m0   = minfo[i][0]
        m01  = minfo[i][1]
        m02  = minfo[i][2]      
        name = 'funsistem'+str(int(binfo[i,0]))    
        res  = int2dtype1(x1,x2,y1,y2,m0,m01,m02,rval,pontosx,pontosy,pesosx,pesosy,eval(name))
        
        bsis[i,0] = res
    
        for j in range(0,nvarsis):
   
           m1   = minfo[j][0]
           m11  = minfo[j][1]
           m12  = minfo[j][2]
           name = 'funsistem'+str(int(ainfo[i,j]))
           res  = int2dtype2(x1,x2,y1,y2,m0,m01,m02,m1,m11,m12,rval,pontosx,pontosy,pesosx,pesosy,eval(name))
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