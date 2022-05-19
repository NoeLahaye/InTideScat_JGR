from __future__ import print_function, division
import time
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
from netCDF4 import Dataset
from get_modes_interp_file import get_modes_interp_file
from R_smooth import get_tri_coef
import R_tools_fort as toolsf

filename = "/net/ruchba/local/tmp/2/lahaye/luckyt_tseries_lf/luckyt_subsamp_stratif.nc"
grdfile = "/net/ruchba/local/tmp/2/lahaye/prep_LUCKYTO/lucky_corgrd.nc"
tserfil = "/net/krypton/data0/project/vortex/lahaye/luckyt_tseries/luckyt_tseries_2Dvars.nc"

imin, imax = 200, 250
jmin, jmax = 800, 845

nc = Dataset(grdfile,'r')
indx = slice(imin,imax)
indy = slice(jmin,jmax)
xx = nc.variables['lon_rho'][indy,indx].T
yy = nc.variables['lat_rho'][indy,indx].T
Nx, Ny = xx.shape
topo = nc.variables['h'][indy,indx].T
nc.close()

nc = Dataset(tserfil)
hc = nc.hc
Cs_r = nc.Cs_r
Cs_w = nc.Cs_w
nc.close()

zr, zw = toolsf.zlevs(topo,np.zeros(topo.shape),hc,Cs_r,Cs_w)

nc = Dataset(filename)
xi = nc.variables['xi_rho'][:]
i1 = np.where(xi>imin)[0][0]-1
i2 = np.where(xi<imax)[0][-1]+2
xi = xi[i1:i2]
eta = nc.variables['eta_rho'][:]
j1 = np.where(eta>jmin)[0][0]-1
j2 = np.where(eta<jmax)[0][-1]+2
eta = eta[i1:i2]
lon = nc.variables['lon_rho'][j1:j2,i1:i2].T
lat = nc.variables['lat_rho'][j1:j2,i1:i2].T
Nysub, Nxsub = nc.variables['h'].shape
#lon = nc.variables['lon_rho'][:].T
#lat = nc.variables['lat_rho'][:].T
nc.close()

# bilinear interpolation using delaunay triangulation
tmes, tmeb = time.clock(), time.time()
elem, coef = get_tri_coef(lon,lat,xx,yy)
eli, elj = np.unravel_index(elem,lon.shape)
eli += i1
elj += j1
gelem = np.ravel_multi_index((eli,elj),(Nxsub,Nysub)) # elems indices in entire grid
print("triangulation",time.clock()-tmes,time.time()-tmeb)

# call it !
(wmodeb, pmodeb), (cmap, norm_w, norm_p), (Nsqb, bbase) = \
    get_modes_interp_file(filename,0,5,gelem,coef,zw,zr,mask=None,doverb=True)

