''' extract_smoothed_strat.py
extract time-space smoothed stratification (lowpass time-filtered, horizontal blur)
this is a try, not done yet
NJAL February 2018 '''
from __future__ import print_function, division
import numpy as np
#from netCDF4 import Dataset
from Modules import *
#import scipy.signal as sig
import itertools as itool
from def_chunk import get_indchk
import comp_zlevs


npx, npy = 16, 16       # number of chunks (1 thread / chunk)
ipx, ipy = 4, 4         # 2D chunk number
simulname = 'lucky0'    # name of simulation
ideb = 10               # starting time step

# horizontal smoothing and subsampling: parameters
nsx = 10        # horizontal subsampling step
sigx = 4.       # horizontal gaussian smooth std
nxsm = 10       # number of points used for smoothing (each side)

varname = 'bvf'

simul = load(simulname,time=ideb)
Nx, Ny = simul.topo.shape

(indx, indy), (nx,ny) = get_indchk((Nx,Ny),(npx,npy),nsx)
ix1 = max(indx[ipx]-nxsm,0)
dbx = indx[ipx]-ix1
ix2 = min(indx[ipx+1]+nxsm,Nx)
iy1 = max(indy[ipy]-nxsm,0)
dby = indy[ipy]-iy1
iy2 = min(indy[ipy+1]+nxsm,Ny)
nx, ny = nx[ipx], ny[ipy]
nxc, nyc = ix2-ix1, iy2-iy1

# define weights and indices (nx*ny*npts array)
topo = simul.topo[ix1:ix2,iy1:iy2]
zz = -comp_zlevs.zlev_rho(topo, np.zeros(nxc,nyc), simul.hc, simul.Cs_r)
zb = zz[dbx::nsx,dby::nsx,:][:nx,:ny]
for ii,jj in itool.product(range(nx),range(ny)):
    slix = slice(dbx+ii*nsx-nxsm,dbx+ii*nsx+nxsm)
    sliy = slice(dby+jj*nsx-nxsm,dby+jj*nsx+nxsm)
    zdif = zz[slix,sliy,:,None] - zb[ii:ii+1,jj:jj+1,None,:]
    iz1 = np.argmin(abs(zdif),axis=-1)
    l,m,n = iz1.shape   # TODO fancy indexing to vectorize the following
    iz2 = np.zeros((l,m,n))
    for i,j,k in products(l,m,n):
        iz2[i,j,k] = iz1[i,j,k] + np.sign(zdif[i,j,k,iz1[i,j,k]])
del zz

#fig, axs = plt.subplots(1,2)
#axs[0].pcolormesh(
