# coding: utf-8
""" test_project_zeta
# ### reconstruct SSH from modal decomposition
# 
# * Incremental: add up modes from 0 to Nmodes, compute error for each
# * one single time step
# * zeta is taken from pressure at the surface
NJAL April 2018 """

from __future__ import print_function, division
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
from netCDF4 import Dataset
from Modules import *

simulpath = "LUCKYM2"
simulname = simulpath.lower()

filepath = '/'.join([ os.getenv('STOREDIR'), simulpath \
                    , 'data/modemap/{}_modemap.nc'.format(simulname) ])
it = 200
kind = ['full','bclin'] # full: sum with mode 0 / bclin: sum modes > 0
dirpic = "/".join([os.getenv('WORKDIR'), simulpath \
                  , 'pictures/modal_mapping/test_SSE/'])
dosavefig = True

grav = 9.81

# plotting parameters
fs       = 12
proj     = 'lcc'
res      = 'i'
stride   = 5
Lx,Ly    = 1500e3,1500e3 # extend in m
cmap = plt.get_cmap('seismic')
zlevs=[0,2000,3500]
grdcol = "grey"
topocol = "grey"

nc = Dataset(filepath,"r")
pamp = nc.variables['p_amp'][it,...]
pmod = nc.variables['p_modes'][:,-1,...]
lon = nc.variables['lon_rho'][:]
lat = nc.variables['lat_rho'][:]
topo = nc.variables['topo'][:]
time = nc.variables['time'][it]
otime = nc.variables['ocean_time'][it]
xi = nc.variables['xi_rho'][:]
eta = nc.variables['eta_rho'][:]
nc.close()

simul = load(simulname,time=time)
zeta = var('zeta',simul).data[xi,:][:,eta].T

## Prep. the plot
fig = plt.figure(figsize=(8,7))
ax = plt.gca()
bm = Basemap(projection=proj,resolution=res,lon_0=lon.mean(),
        lat_0=lat.mean(),width=Lx,height=Ly)
xx, yy = bm(lon, lat)
bm.drawcoastlines(color='gray')
bm.fillcontinents(color='gray')
bm.drawparallels(np.arange(-60,70,stride),labels=[1,0,0,0],linewidth=0.8,                fontsize=fs,color=grdcol)
bm.drawmeridians(np.arange(-100,100,stride),labels=[0,0,0,1],linewidth=0.8,                fontsize=fs,color=grdcol)
hct = bm.contour(xx,yy,topo,levels=zlevs,colors=topocol,linewidths=1,alpha=0.8)
for item in hct.collections:
    item.set_rasterized(True)

##### plot the data
# SSE
toplot = zeta.copy()
hmoy = np.nanmean(toplot)
toplot -= hmoy
valmax = 2*np.nanstd(toplot)

hpc = bm.pcolormesh(xx,yy,toplot,vmin=-valmax,vmax=valmax,cmap=cmap)
hcb = bm.colorbar(hpc,extend='both')

hcb.formatter.set_powerlimits((-1, 1))    
hcb.update_ticks()
hcb.ax.tick_params(labelsize=fs)

plt.title('SSE at t={0} -- mean = {1:.2f}'.format(int(otime),hmoy))
if dosavefig:
    plt.savefig(dirpic+simulname+'_SSE_tot_t{:04d}.png'.format(int(otime)) \
                , magnification="auto", dpi=200, bbox_inches='tight')

# mod 0 for SSE
toplot = pamp[0,...]*pmod[0,...]/grav
hmoy = np.nanmean(toplot)
toplot -= hmoy
hpc.set_array(toplot[:-1,:-1].ravel())
valmax = 2*np.nanstd(toplot)
hpc.set_clim(np.array([-1,1])*valmax)

plt.title('SSE(n=0) at t={0} -- mean = {1:.2f}'.format(int(otime),hmoy))
if dosavefig:
    plt.savefig(dirpic+simulname+'_SSEtot_mod00_t{:04d}.png'.format(int(otime)) \
                , magnification="auto", dpi=200, bbox_inches='tight')

### plot the error
toplot -= zeta - hmoy
hpc.set_array(toplot[:-1,:-1].ravel())
valmax = 2*np.nanstd(toplot)
hpc.set_clim(np.array([-1,1])*valmax)
plt.title('SSE-SSE(n=0) at t={0} -- mean/std: {1:.2f} / {2:.2e}'.format(int(otime) \
                , hmoy, np.nanstd(toplot)))
if dosavefig:
    plt.savefig(dirpic+simulname+'_SSEtot-error_mod00_t{:04d}.png'.format(int(otime)) \
                , magnification="auto", dpi=200, bbox_inches='tight')
print("done with mode 0")

for imod in range(1,pmod.shape[0]):
    if "full" in kind:
        # total signal, modal truncated reconstruction
        toplot = np.nansum(pmod[:imod+1,...]*pamp[:imod+1,...],axis=0)/grav
        hmoy = np.nanmean(toplot)
        toplot -= hmoy
        hpc.set_array(toplot[:-1,:-1].ravel())
        valmax = 2*np.nanstd(toplot)
        hpc.set_clim(np.array([-1,1])*valmax)
        plt.title('Total SSE(n={0}) at t={1} -- mean={2:.2f}'.format(int(imod) \
                                    , int(otime), hmoy))
        if dosavefig:
            plt.savefig(dirpic+simulname+'_SSEtot_mod{0:02d}_t{1:04d}.png'.format(imod \
                        , int(otime)), magnification='auto', dpi=200, bbox_inches="tight")
        # total signal, error of truncation
        toplot -= zeta - hmoy
        hpc.set_array(toplot[:-1,:-1].ravel())
        valmax = 2*np.nanstd(toplot)
        hpc.set_clim(np.array([-1,1])*valmax)
        plt.title( \
                'Total SSE-SSE(n={0}) at t={1} -- mean/std: {2:.2f} / {3:.2e}'\
                    .format(int(imod), int(otime), hmoy, valmax/2.))
        if dosavefig:
            plt.savefig(dirpic+simulname+'_SSEtot-error_mod{0:02d}_t{1:04d}.png'.format(imod \
                        , int(otime)), magnification='auto', dpi=200, bbox_inches="tight")
    if "bclin" in kind:
        # baroclinic signal, modal truncated reconstruction
        toplot = np.nansum(pmod[1:imod+1,...]*pamp[1:imod+1,...],axis=0)/grav
        hmoy = np.nanmean(toplot)
        toplot -= hmoy
        hpc.set_array(toplot[:-1,:-1].ravel())
        valmax = 2*np.nanstd(toplot)
        hpc.set_clim(np.array([-1,1])*valmax)
        plt.title('Int. SSE(n={0}) at t={1} -- mean={2:.2f}'.format(int(imod) \
                                    , int(otime), hmoy))
        if dosavefig:
            plt.savefig(dirpic+simulname+'_SSEint_mod{0:02d}_t{1:04d}.png'.format(imod \
                        , int(otime)), magnification='auto', dpi=200, bbox_inches="tight")
        # total signal, error of truncation
        toplot -= zeta - pmod[0,...]*pamp[0,...]/grav - hmoy
        hpc.set_array(toplot[:-1,:-1].ravel())
        valmax = 2*np.nanstd(toplot)
        hpc.set_clim(np.array([-1,1])*valmax)
        plt.title(\
            'Int. SSE-SSE(n={0}) at t={1} -- mean/std: {2:.2f},{3:.2e}'.\
                format(int(imod), int(otime), hmoy, valmax/2.))
        if dosavefig:
            plt.savefig(dirpic+simulname+'_SSEint-error_mod{0:02d}_t{1:04d}.png'.format(imod \
                        , int(otime)), magnification='auto', dpi=200, bbox_inches="tight")
    print("done with mode",imod)
