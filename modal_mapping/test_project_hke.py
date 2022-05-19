# coding: utf-8
""" test_project_hke
# ### reconstruct surface HKE from modal decomposition
# 
# * Incremental: add up modes from 0 to Nmodes, compute error for each
# * one single time step
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
dosavefig = True
if dosavefig:
    dirpic = "/".join([os.getenv('WORKDIR'), simulpath \
                      , 'pictures/modal_mapping/test_HKE/'])
    if not os.path.isdir(dirpic):
        os.mkdir(dirpic)

rho0 = 1.025    # HKE in kJ/m^3

# plotting parameters
fs       = 12
proj     = 'lcc'
res      = 'i'
stride   = 5
Lx,Ly    = 1500e3,1500e3 # extend in m
cmap_err = plt.get_cmap('seismic')
cmap_abs = plt.get_cmap('gist_stern_r')
zlevs=[0,2000,3500]
grdcol = "grey"
topocol = "grey"

nc = Dataset(filepath,"r")
uamp = nc.variables['u_amp'][it,...]
vamp = nc.variables['v_amp'][it,...]
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
uu = tools.u2rho(var('u',simul,depths=[0]).data)[xi,:][:,eta].T
vv = tools.v2rho(var('v',simul,depths=[0]).data)[xi,:][:,eta].T
hke = rho0/2. * (uu**2 + vv**2) 

### Prep. the plot
fig = plt.figure(figsize=(8,7))
ax = plt.gca()
bm = Basemap(projection=proj,resolution=res,lon_0=lon.mean(),
        lat_0=lat.mean(),width=Lx,height=Ly)
xx, yy = bm(lon, lat)
bm.drawcoastlines(color='gray')
bm.fillcontinents(color='gray')
bm.drawparallels(np.arange(-60,70,stride),labels=[1,0,0,0],linewidth=0.8,fontsize=fs,color=grdcol)
bm.drawmeridians(np.arange(-100,100,stride),labels=[0,0,0,1],linewidth=0.8,fontsize=fs,color=grdcol)
hct = bm.contour(xx,yy,topo,levels=zlevs,colors=topocol,linewidths=1,alpha=0.8)
for item in hct.collections:
    item.set_rasterized(True)

##### plot the data
# HKE
toplot = hke.copy()
hmoy = np.nanmean(toplot)

hpc = bm.pcolormesh(xx,yy,toplot,vmin=0,vmax=2*hke.std(),cmap=cmap_abs)
hcb = bm.colorbar(hpc) #,extend='max')

hcb.formatter.set_powerlimits((-1, 1))    
hcb.update_ticks()
hcb.ax.tick_params(labelsize=fs)

plt.title('surf. HKE at t={0} -- mean = {1:.2e}'.format(int(otime),hmoy))
if dosavefig:
    plt.savefig(dirpic+simulname+'_sHKE_total_t{:04d}.png'.format(int(otime)) \
                , magnification="auto", dpi=200, bbox_inches='tight')

# HKE mode 0
toplot = ( (uamp[0,...]*pmod[0,...])**2 + (vamp[0,...]*pmod[0,...])**2)*rho0/2.
hke0 = toplot.copy()
hmoy = np.nanmean(toplot)
hpc.set_array(toplot[:-1,:-1].ravel())
valmax = 2*np.nanstd(toplot)
hpc.set_clim(np.array([0,1])*valmax)

plt.title('surf. HKE(n=0) at t={0} -- mean = {1:.2e}'.format(int(otime),hmoy))
if dosavefig:
    plt.savefig(dirpic+simulname+'_sHKE_mod00_t{:04d}.png'.format(int(otime)) \
                , magnification="auto", dpi=200, bbox_inches='tight')

### plot the error
toplot = hke - toplot
hpc.set_array(toplot[:-1,:-1].ravel())
valmax = 2*np.nanstd(toplot)
hpc.set_clim(np.array([-1,1])*valmax)
hpc.set_cmap(cmap_err)
plt.title('sHKE-sHKE(n=0) at t={0} -- mean/std: {1:.2e} / {2:.2e}'.format(int(otime) \
                , hmoy, valmax/2.))
if dosavefig:
    plt.savefig(dirpic+simulname+'_sHKEtot-error_mod00_t{:04d}.png'.format(int(otime)) \
                , magnification="auto", dpi=200, bbox_inches='tight')

print("done with mode 0")

for imod in range(1,pmod.shape[0]):
    if "full" in kind:
        # total signal, modal truncated reconstruction
        toplot = ( np.nansum(uamp[:imod+1,...]*pmod[:imod+1,...],axis=0)**2 \
                  + np.nansum(vamp[:imod+1,...]*pmod[:imod+1,...],axis=0)**2)*rho0/2.
        hmoy = np.nanmean(toplot)
        hpc.set_array(toplot[:-1,:-1].ravel())
        hpc.set_cmap(cmap_abs)
        valmax = 2*np.nanstd(toplot)
        hpc.set_clim(np.array([0,valmax]))
        plt.title('Total sHKE(n={0}) at t={1} -- mean={2:.2e}'.format(int(imod) \
                                    , int(otime), hmoy))
        if dosavefig:
            plt.savefig(dirpic+simulname+'_sHKEtot_mod{0:02d}_t{1:04d}.png'.format(imod \
                        , int(otime)), magnification='auto', dpi=200, bbox_inches="tight")
        # total signal, error of truncation
        toplot = hke - toplot
        hpc.set_array(toplot[:-1,:-1].ravel())
        valmax = 2*np.nanstd(toplot)
        hpc.set_clim(np.array([-1,1])*valmax)
        hpc.set_cmap(cmap_err)
        plt.title('Total sHKE-sHKE(n={0}) at t={1} -- mean/std: {2:.2e} / {3:.2e}'.format( \
                        int(imod), int(otime), hmoy, valmax/2.))
        if dosavefig:
            plt.savefig(dirpic+simulname+'_sHKEtot_error_mod{0:02d}_t{1:04d}.png'.format(imod \
                        , int(otime)), magnification='auto', dpi=200, bbox_inches="tight")
    if "bclin" in kind:
        # baroclinic signal, modal truncated reconstruction
        toplot = ( np.nansum(uamp[1:imod+1,...]*pmod[1:imod+1,...],axis=0)**2 \
                  + np.nansum(vamp[1:imod+1,...]*pmod[1:imod+1,...],axis=0)**2)*rho0/2.
        hmoy = np.nanmean(toplot)
        valmax = 2*np.nanstd(toplot)
        hpc.set_array(toplot[:-1,:-1].ravel())
        hpc.set_cmap(cmap_abs)
        hpc.set_clim(np.array([0,1])*valmax)
        plt.title('Int. sHKE(n={0}) at t={1} -- mean={2:.2e}'.format(int(imod) \
                                    , int(otime), hmoy))
        if dosavefig:
            plt.savefig(dirpic+simulname+'_sHKEint_mod{0:02d}_t{1:04d}.png'.format(imod \
                        , int(otime)), magnification='auto', dpi=200, bbox_inches="tight")
        # total signal, error of truncation
        toplot = hke - hke0 - toplot
        hpc.set_array(toplot[:-1,:-1].ravel())
        valmax = 2*np.nanstd(toplot)
        hpc.set_cmap(cmap_err)
        hpc.set_clim(np.array([-1,1])*valmax)
        plt.title('Int. sHKE-sHKE(n={0}) at t={1} -- mean/std: {2:.2e} / {3:.2e}'.format(\
                        int(imod), int(otime), hmoy, valmax/2.))
        if dosavefig:
            plt.savefig(dirpic+simulname+'_sHKEint_error_mod{0:02d}_t{1:04d}.png'.format(imod \
                        , int(otime)), magnification='auto', dpi=200, bbox_inches="tight")
    print("done with mode",imod)
