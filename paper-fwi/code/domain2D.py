#==============================================================================
# Devito Imports
#==============================================================================
from   devito import *
#==============================================================================
import numpy                   as np
#==============================================================================

#==============================================================================
# Subdomain D0
#==============================================================================
class physdomain(SubDomain):
    def __init__(self, npmlx,npmlz):    
        self.npmlx=npmlx
        self.npmlz=npmlz
    name = 'd0'
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('middle', self.npmlx, self.npmlx), z: ('middle', 0, self.npmlz)}
#==============================================================================

#==============================================================================
# Subdomain D1
#==============================================================================
class leftExtension(SubDomain):
    def __init__(self, npmlx,npmlz):    
        self.npmlx=npmlx
        self.npmlz=npmlz
    name = 'd1'
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('left',self.npmlx), z: z}
#==============================================================================

#==============================================================================
# Subdomain D2
#==============================================================================
class rightExtension(SubDomain):
    def __init__(self, npmlx,npmlz):    
        self.npmlx=npmlx
        self.npmlz=npmlz
    name = 'd2'
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('right',self.npmlx), z: z}
#==============================================================================

#==============================================================================
# Subdomain D3
#==============================================================================
class bottomExtension(SubDomain):
    def __init__(self, npmlx,npmlz):    
        self.npmlx=npmlx
        self.npmlz=npmlz
    name = 'd3'
    def define(self, dimensions):
        x, z = dimensions
        return {x: x, z: ('right',self.npmlz)}
#==============================================================================

#==============================================================================
# Domain and Subdomains Settings
#==============================================================================  
def SetGrid(setup):   
    
    d0_domain = physdomain(setup.npmlx,setup.npmlz)
    d1_domain = leftExtension(setup.npmlx,setup.npmlz)
    d2_domain = rightExtension(setup.npmlx,setup.npmlz)
    d3_domain = bottomExtension(setup.npmlx,setup.npmlz)

    origin  = (setup.x0, setup.z0)
    extent  = (setup.compx,setup.compz)
    shape   = (setup.nptx,setup.nptz)
    grid    = Grid(origin=origin,extent=extent,shape=shape,subdomains=(d0_domain,d1_domain,d2_domain,d3_domain),dtype=np.float64) 
    
    return grid
#==============================================================================