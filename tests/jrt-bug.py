import os
import sys
import numpy as np
import argparse
from examples.seismic import Model
from examples.seismic import TimeAxis
from sympy import cos, sin
#from devito import cos, sin
from devito import Eq, Grid, solve
from devito import TimeFunction
from devito import centered, first_derivative, second_derivative, right, transpose
from devito import SparseTimeFunction
from devito import Operator
from devito import configuration
from devito import norm
from matplotlib import pyplot
from mpi4py import MPI

def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1 - 2.*r**2) * np.exp(-r**2)

def tti_operator(model, u, v, time_order=2, space_order=2, dt=.01, nt=10, save=False, **kwargs):

    ang0 = cos(model.theta)
    ang1 = sin(model.theta)
    ang2 = cos(model.phi)
    ang3 = sin(model.phi)
            
    
    H1u=np.square(ang1)*np.square(ang2)*u.dx2 + np.square(ang1)*np.square(ang3)*u.dy2 + np.square(ang0)*u.dz2 + np.square(ang1)*2*ang3*ang2*u.dxdy + (2*ang0*ang1)*ang3*u.dydz + 2*ang0*ang1*ang2*u.dxdz
    H2u = -H1u + u.laplace
    
    H1v=np.square(ang1)*np.square(ang2)*v.dx2 + np.square(ang1)*np.square(ang3)*v.dy2 + np.square(ang0)*v.dz2 + np.square(ang1)*2*ang3*ang2*v.dxdy + 2*ang0*ang1*ang3*v.dydz + 2*ang0*ang1*ang2*v.dxdz
    H2v = -H1v + v.laplace
 

    
    # Derive stencil from symbolic equation
    stencilp =  (2*u - u.backward) +  dt**2/model.m * (model.epsilon * H2u + H1v)  
    stencilr =  (2*v - v.backward) +  dt**2/model.m * (model.delta * H2u + H1v)
    

    #this causes no issues
    #stencilp =  (2*u - u.backward) +  dt**2/model.m * (u.laplace) 
    #stencilr =  (2*v - v.backward) +  dt**2/model.m * (v.laplace)
    
    
    #this also causes no issues
    #stencilp =  (2*u - u.backward) +  dt**2/model.m * (v.laplace) 
    #stencilr =  (2*v - v.backward) +  dt**2/model.m * (u.laplace)

    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    stencils = [first_stencil, second_stencil]
    return stencils

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Input parameters')
    parser.add_argument("--nx", default=64, help="number of points in x dimension", type=int)
    parser.add_argument("--ny", default=64, help="number of points in y dimension", type=int)
    parser.add_argument("--nz", default=64, help="number of points in z dimension", type=int)
    parser.add_argument("--radius", default=2, help="radius of stencil", type=int)
    args=parser.parse_args()

    
    
    configuration['autotuning']='off'
    configuration['dse']='advanced' ##turning this off there's almost 0 issue
    configuration['log-level']='DEBUG'
    
    
    #configuration['openmp']=True
    #configuration['openmp']=False ##this also fails, just takes longer
    
    #configuration['mpi']='diag'
    
    radius = args.radius
    space_order=radius*2
    time_order=2
    nbpml=0
    nx, ny, nz = args.nx, args.ny, args.nz
    freq= 15.0
    Np = 10
    c = 3000
    h = c/(Np*freq)
    dx, dy, dz = h, h, h
    dt = 1.0/(10.0)*h/c
    nt = 200
    dimensions=(nx,ny,nz)
    spacing=(dx,dy,dz)
    origin=(0.,0.,0.)

    
    
    
    
    
    true_vp=np.ones(dimensions)*2500.0
 

    model=Model(vp=true_vp, origin=origin, shape=dimensions, spacing=spacing, space_order=space_order, nbpml=nbpml, epsilon=.2*np.ones(dimensions), delta=.1*np.ones(dimensions), theta=np.pi/4.0*np.ones(dimensions), phi=np.pi/4.0*np.ones(dimensions) ) 
    
    comm=model.grid.distributor.comm
    rank=comm.Get_rank()
    size=comm.Get_size()

    u=TimeFunction(name='u', grid=model.grid, time_order=time_order, space_order=space_order)
    v=TimeFunction(name='v', grid=model.grid, time_order=time_order, space_order=space_order)
    stencils=tti_operator(model, u, v, time_order=time_order, space_order=space_order, dt=dt, nt=nt, save=False)
    
    t0=0
    tn=dt*nt
    time_vals= np.linspace(0, tn, num=nt)
    
    
    src_location=np.array(model.domain_size)*.5 #this causes some issues with mpi ranks = e.g. 4 since halos occur near injection
    #src_location=np.array(model.domain_size)*.0 #this causes no issues with mpi 
    
    src_vals=source(time_vals, freq)
    sf = SparseTimeFunction(name='sf', grid=model.grid, npoint=1, nt=nt, coordinates=src_location)
    sf.data[:][:,0]=src_vals[:] #sf.data is a recarray, sf.data[:] returns point to numpy array of size (nt, npoints), the number of injections. sf.data[:][:,0] references an array in time of values of a single source injection at one physical coordinate
    
    exprs1 = sf.inject(u.forward,expr=sf*dt*dt/model.m)
    exprs2 = sf.inject(v.forward,expr=sf*dt*dt/model.m)
    u.data[:]=0.0
    v.data[:]=0.0



    op=Operator(stencils+exprs1+exprs2) #race condition dependent on stencil, src injection location
    
    kwargs={'time_M':nt-1}
    summary= op.apply(**kwargs) 

    

    normu=norm(u)
    normv=norm(v)
    sys.stdout.flush()
    comm.Barrier()
   

    if rank==0:
        print("Norm of u is : %f " % normu)  
        print("Norm of v is : %f " % normv)  
    
    
    sys.stdout.flush()
    comm.Barrier()
