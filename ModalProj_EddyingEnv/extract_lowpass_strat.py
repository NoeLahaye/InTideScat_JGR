''' extract_lowpass_strat.py
extract lowpass time filtered stratification 
this is a try, not done yet
NJAL February 2018 '''
from __future__ import print_function, division
import numpy as np
#from netCDF4 import Dataset
from Modules import *
import scipy.signal as sig
import itertools as itool
from def_chunk import get_indchk
import comp_zlevs


npx, npy = 4, 4         # number of chunks (1 thread / chunk)
ipx, ipy = 2, 2         # 2D chunk number
simulname = 'lucky0'    # name of simulation
ideb = 10               # starting time step
iend = 100
nsx = 10        # horizontal subsampling step
nst = 24        # subsampling time step
filename = "/net/ruchba/local/tmp/2/lahaye/smoothed_strat/{}_lowf_stratif.nc".format(simulname)

varname = ['buoy']
fcut = 1./30    # lowpass filter cut-off frequency (1/h)

simul = load(simulname,time=ideb)
Nx, Ny = simul.topo.shape

(indx, indy), (nx,ny) = get_indchk((Nx,Ny),(npx,npy),nsx)
ix1, ix2 = indx[ipx:ipx+2]
iy1, iy2 = indy[ipy:ipy+2]
nx, ny = nx[ipx], ny[ipy]
nxc, nyc = ix2-ix1, iy2-iy1
indt, nt = get_indchk(iend-ideb,1,nst)  # just to get indices (no chunking)

# extract grid
topo = simul.topo[ix1:ix2:nsx,iy1:iy2:nsx]
zz = -comp_zlevs.zlev_w(topo, np.zeros(nx,ny), simul.hc, simul.Cs_w)
dt = simul.dt/3600.
bb, aa = sig.butter(4,fcut*2*dt,btype="lowpass")
nz = zz.shape[-1]

# create netCDF file
ncw = Dataset(filename,'w')
ncw.createDimension('time',None)
ncw.createDimension('xi_rho',nx)
ncw.createDimension('eta_rho',ny)
ncw.createDimension('s_w',nz)
for dim in ncw.dimensions.values():
    ncw.createVariable(dim.name,'i',(dim.name,))
ncw.variables['xi_rho'][:] = np.arange(indx[0],Nx,nsx)
ncw.variables['eta_rho'][:] = np.arange(indy[0],Ny,nsx)
ncw.variables['s_w'][:] = np.arange(nz)

for it in range(ideb,iend):
    data = var(varname,simul,coord=[iy1,iy2,ix1,ix2],depth)

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
