from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
plt.ion()
from Modules import *
from get_pres import get_redpres
from def_chunk import get_indchk
from get_modes_interp_file import get_modes_interp_file
import comp_zlevs
import time, os, datetime, sys

simulname = 'luckyt'   # name of simulation
sidirname = simulname.upper()
Nmodes = 2             # number of vertical modes to use
nst = 4                 # horizontal subsampling
fildir = '/ccc/scratch/cont003/gen7638/lahayen/{}/data/'.format(sidirname)
bastate_filename = fildir+"tseries/{}_subsamp_stratif.nc".format(simulname)
ideb = 5549 #2705       # it start computing modal amplitude

npy, npx = (8,8)        # number of chunks in y, x (mpi)
subpx = range(npx)  # put None or range(npx) to do every chunks
subpy = range(npy)  # (put something else e.g. when appending)
ipx, ipy = 0, 0

grav = 9.81
rho0 = 1025.0

#####   load simulation, deal with chunking
simul = load(simulname,time=0)
Nx, Ny = simul.topo.shape
Nz = simul.Cs_r.__len__()
(indx,indy), (nxs,nys) = get_indchk((Nx-2,Ny-2),(npx,npy),nst)
indx = [i+1 for i in indx] 
indy = [i+1 for i in indy]
nx, ny = nxs[ipx], nys[ipy]
slix = slice(indx[ipx],indx[ipx+1],nst)
sliy = slice(indy[ipy],indy[ipy+1],nst)
topo = simul.topo[slix,sliy]
mask = simul.mask[slix,sliy]
coords = [indy[ipy],indy[ipy+1],indx[ipx],indx[ipx+1]]
coordu = [indy[ipy],indy[ipy+1],indx[ipx]-1,indx[ipx+1]+1]
coordv = [indy[ipy]-1,indy[ipy+1]+1,indx[ipx],indx[ipx+1]]

nc = Dataset(bastate_filename)
xi = nc.variables['xi_rho'][:]
i1 = np.where(xi>indx[ipx])[0][0]-1
i2 = np.where(xi<indx[ipx+1])[0][-1]+2
eta = nc.variables['eta_rho'][:]
j1 = np.where(eta>indy[ipy])[0][0]-1
j2 = np.where(eta<indy[ipy+1])[0][-1]+2
lon = nc.variables['lon_rho'][j1:j2,i1:i2].T
lat = nc.variables['lat_rho'][j1:j2,i1:i2].T
Nysub, Nxsub = nc.variables['h'].shape
times = nc.variables['time'][:]
nc.close()
tmes, tmeb = time.clock(), time.time()
elem, coef = sm.get_tri_coef(lon,lat,simul.x[slix,sliy],simul.y[slix,sliy])
eli, elj = np.unravel_index(elem,lon.shape)
elem = np.ravel_multi_index((eli+i1,elj+j1),(Nxsub,Nysub)) # elems indices in entire grid
print("triangulation",time.clock()-tmes,time.time()-tmeb)
xi, eta = xi[i1:i2], eta[j1:j2]
nbx, nby = xi.size, eta.size

zr, zw = toolsF.zlevs(simul.topo[slix,sliy], np.zeros((nx,ny)), simul.hc, simul.Cs_r \
                        , simul.Cs_w)   # z-grid at computed points in chunk
nmodes = Nmodes

# First computation of modes
it0 = np.where(times <= ideb)[0][-1]
(wmodeb, pmodeb), (cmap, norm_w, norm_p), (Nsqb, bbase) = \
        get_modes_interp_file(bastate_filename, it0, Nmodes \
                        , elem, coef, zw, zr, mask, doverb=True )

nsqb = Nsqb[25,57:59,:].squeeze()
bbas = bbase[25,57:59,:].squeeze()

zr = zr[25,57:59,:].squeeze()
zw = zw[25,57:59,:].squeeze()

#plt.figure(1)
#plt.plot(nsqb.T,zw.T)
#plt.grid(True)

plt.figure(2)
plt.plot(bbas.T,zr.T)
plt.grid(True)
