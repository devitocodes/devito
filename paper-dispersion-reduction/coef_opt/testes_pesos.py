#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as     np
import sys
from   scipy.integrate         import quad
import matplotlib.pyplot       as plot
import time                    as tm
#==============================================================================

#==============================================================================
# Yang Liu 2012 Coefficients
#==============================================================================
sys.path.insert(0, './coef_yang_liu')
import coef_otm_versao11        as     cotm11
import coef_otm_versao12        as     cotm12
import coef_otm_versao13        as     cotm13
import coef_otm_versao14        as     cotm14
#==============================================================================

#==============================================================================
# Li-Liu-Ren-Sen 2017 Coefficients
#==============================================================================
sys.path.insert(0, './coef_li_liu_ren_sen')
import coef_otm_versao21        as     cotm21
#==============================================================================

#==============================================================================
# Liu and Sen 2009 Coefficients
#==============================================================================
sys.path.insert(0, './coef_liu_sen')
import coef_otm_versao31        as     cotm31
import coef_otm_versao32        as     cotm32
#==============================================================================

#==============================================================================
# Yajun 2016 Coefficients
#==============================================================================
sys.path.insert(0, './coef_yajun')
import coef_otm_versao41        as     cotm41
import coef_otm_versao42        as     cotm42
#==============================================================================

#==============================================================================
# Kastner and Finkelstein 2006 Coefficientes
#==============================================================================
sys.path.insert(0, './coef_kastner_finkelstein')
import coef_otm_versao51        as     cotm51
import coef_otm_versao52        as     cotm52
#==============================================================================

#==============================================================================
# Edward Caunt - Master Dissertations Coefficients
#==============================================================================
sys.path.insert(0, './coef_edc')
import coef_otm_versao61        as     cotm61
#==============================================================================

#==============================================================================
# Wang, Liu and Sen - Cross/Rombus Stencils - 2016
#==============================================================================
sys.path.insert(0, './coef_wang_liu_sen')
import coef_otm_versao71        as     cotm71
import coef_otm_versao72        as     cotm72
import coef_otm_versao721       as     cotm721
import coef_otm_versao722       as     cotm722
import coef_otm_versao723       as     cotm723
#==============================================================================

#==============================================================================
plot.close("all")
#==============================================================================

#==============================================================================
# T1 - Yang Liu 2012 Coefficients - FD coefficients calculated by LS
#==============================================================================
#nordem = 10
#bint   = 2.45
#T1 = cotm11.calccoef(nordem,bint)
#==============================================================================

#==============================================================================
# T2 Type - Coef Yang Liu - A globally optimal FD scheme based on space-domain 
# dispersion relation
#==============================================================================
#nordem = 10
#bint   = 2.32
#T2     = cotm12.calccoef(nordem,bint)
#==============================================================================

#==============================================================================
# T3 Type - Coef Yang Liu - A nearly globally optimal FD scheme based on 1D
# time-space-domain dispersion relation
#==============================================================================
#nordem = 8
#bint   = 2.10
#rval   = 0.048
#T3     = cotm13.calccoef(nordem,bint,rval)
#==============================================================================

#==============================================================================
# T4 Type - Coef Yang Liu - A nearly globally optimal FD scheme based on 2D
# time-space-domain dispersion relation
#==============================================================================
#nordem  = 8
#bint    = 2.74
#tetaint = np.pi/4
#rval    = 0.15 
#T4      = cotm14.calccoef(nordem,bint,tetaint,rval)
#==============================================================================

#==============================================================================
# T5 Type - Coef Li, Liu, Sen and Ren - A time-space-domain mesh-free finite
# difference based on least squares
#==============================================================================
#nordem = 4
#K      = np.pi
#x0     = 0
#y0     = 0
#dx     = 1
#dy     = 1
#dt     = 1
#vmax   = 1
#rval   = dt*vmax
#T5xx,T5yy = cotm21.calccoef(nordem,K,x0,y0,dx,dy,rval)
#==============================================================================

#==============================================================================
# T6 Type - Coef Liu and Sen - Time–space domain high-order FD method for 1D 
# acoustic  wave equations 
#==============================================================================
#nordem = 2
#dx     = 1
#dt     = 1
#vmax   = 1
#rvalx  = dt*vmax/dx
#T6     = cotm31.calccoef(nordem,rvalx)
#==============================================================================

#==============================================================================
# T7 Type - Coef Liu and Sen - Time–space domain high-order FD method for 2D 
# acoustic  wave equations
#==============================================================================
#nordem    = 4
#dx        = 20.0
#dt        = 1
#vmax      = 1.5
#rval      = dt*vmax/dx
#tetaval   = np.pi/8
#T7        = cotm32.calccoef(nordem,rval,tetaval)
#==============================================================================

#==============================================================================
# T8 Type- Coef Yajun - Uniform dispersion reduction schemes
#==============================================================================
#nordem  = 2
#knumber = 2
#dx      = 1
#dt      = 1
#vmax    = 1
#rval    = dt*vmax/dx
#rval    = 0.4
#T8      = cotm41.calccoef(nordem,rval,knumber)
#==============================================================================

#==============================================================================
# T9 Type - Kastner and Finkelstein 2007-2008
#==============================================================================
#nordem  = 2
#knumber = 2
#dx      = 1
#dt      = 1
#vmax    = 1
#rval    = dt*vmax/dx
#rval    = 0.4
#T9      = cotm51.calccoef(nordem,rval,knumber)
#==============================================================================

#==============================================================================
# T10 Type - Kastner e Finkelstein 2007-2008
#==============================================================================
#nordem  = 2
#knumber = 2
#dx      = 1
#dt      = 1
#vmax    = 1
#rval    = dt*vmax/dx
#rval    = 0.4
#T10     = cotm52.calccoef(nordem,rval,knumber)
#==============================================================================

#==============================================================================
# T11 Type - Edward Caunt - Master Dissertations Coefficients
#==============================================================================
#nordem  = 2
#ndim    = 1
#T11     = cotm61.calccoef(nordem,ndim)
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
#mvalue = 2
#nvalue = 1
#dx   = 20.0
#dt   = 1.0
#vmax = 1.5
#rval = (dt*vmax)/dx
#mcoef  = cotm71.calccoef(mvalue,nvalue,rval,rval)  
#==============================================================================

#==============================================================================
# T12 Type - Wang, Liu and Sen - Effective finite-difference modelling 
# methods with 2-D acoustic wave equation using a combination of cross 
# and rhombus stencils
#==============================================================================
mvalue = 5
nvalue = 1
dx   = 0.4
dt   = 0.03
vmax = 1.5
rval = (dt*vmax)/dx

mcoef2  = cotm72.calccoef(mvalue,nvalue,rval,rval)
name2='mcoef2'
np.save(name2,mcoef2)

mcoef21  = cotm721.calccoef(mvalue,nvalue,rval,rval)
name21='mcoef21'
np.save(name21,mcoef21)

mcoef22  = cotm722.calccoef(mvalue,nvalue,rval,rval)
name22='mcoef22'
np.save(name22,mcoef22)

mcoef23  = cotm723.calccoef(mvalue,nvalue,rval,rval)
name23='mcoef23'
np.save(name23,mcoef23)
#==============================================================================