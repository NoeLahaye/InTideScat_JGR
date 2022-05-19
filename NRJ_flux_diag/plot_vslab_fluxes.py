''' plot energy fluxes and energy surface density
adapted from plot_fluxes.py to vertical sections
for simplicity, axes are both in kilometers (and angle of quiver for fluxes does not 
need special treatment)
NJAL June 2017 '''
from __future__ import print_function, division
dosavefig = False
import matplotlib as mpl
if dosavefig: mpl.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
if not dosavefig: plt.ion()
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate as scinterp
import R_smooth as sm

path_base = '/net/krypton/data0/project/vortex/lahaye/luckyto_vslab_ridge/'
path_fig = path_base+'pictures/re_'
fil_flux = path_base+'luckyto_nrj_fluxes.nc'
pictype='.png'
nrjmap = plt.get_cmap('afmhot_r') #'Greys'
qcol = 'blue'
nxq = 40
nzq = 15
nrjamp = 2e-2
quiv_scale = 1e-1

ncvar = Dataset(fil_flux,'r').variables
lslab = ncvar['lslab'][:]
zr = ncvar['z_rho'][0,...].T
nz = zr.shape[0]

xx = np.linspace(lslab[0],lslab[-1],nxq)
zz = np.linspace(-3500,-20,nzq)
#zz = np.linspace(-800,-20,nzq)

topo = Dataset(path_base+'luckyto_vslab_ridge.nc','r').variables['z_w'][0,:,0]
ftopo = scinterp.PchipInterpolator(lslab,topo)
#otime = ncvar['ocean_time'][:]
otime = np.arange(10,720)
Fx = ncvar['pubc_lf'][:]
Fz = ncvar['pw_lf'][:]
#Famp = np.sqrt(Fx**2 + Fz**2)
# plot energy
nrj = ncvar['ek_lf'][:] + ncvar['ep_lf'][:]
axe = [lslab.min()/1e3,lslab.max()/1e3,topo.min()/1e3,0]
nt = otime.size

xxx,zzz = np.meshgrid(xx,zz)
indoz,indox = np.where(zzz>(ftopo(xxx)+30))  # at least one rho-point above bottom
elem,coef = sm.get_tri_coef(np.tile(lslab,(nz,1)),zr,\
        xxx[indoz,indox][:,None],zzz[indoz,indox][:,None])
Fsx = np.ma.masked_invalid(np.full((nzq,nxq),np.nan)); Fsz = Fsx.copy()
Fsx[indoz,indox] = (coef.squeeze()*(Fx[0,...].ravel()[elem.squeeze()])).sum(axis=-1)
Fsz[indoz,indox] = (coef.squeeze()*(Fz[0,...].ravel()[elem.squeeze()])).sum(axis=-1)

fig = plt.figure(figsize=(12,6))
hpc = plt.pcolormesh(lslab[None,:]/1e3,zr/1e3,nrj[0,...],vmin=0,vmax=nrjamp,cmap=nrjmap,shading='gouraud')
#hpc = plt.pcolormesh(lslab[None,:]/1e3,zr/1e3,nrj[0,...],vmin=0,vmax=nrjamp,cmap=nrjmap)
#hpc = plt.contourf(np.tile(lslab/1e3,(nz,1)),zr/1e3,nrj[0,...],vmin=0,vmax=nrjamp,cmap=nrjmap) # does not work
hqv = plt.quiver(xx/1e3,zz/1e3,Fsx,Fsz,pivot='tail',color=qcol,angles='xy',scale=quiv_scale)
plt.ylim([topo.min(),0])
#plt.grid(True)
plt.xlabel('distance (km)')
plt.ylabel('depth (m)')
hcb = fig.colorbar(hpc,extend='max',pad=0.01)
hcb.set_label(r"energy density (J/m$^3$)")
plt.plot(lslab/1e3,topo/1e3,'k',lw=2)
plt.title(r"LUCKYTO: bclin. nrj. density at $t={}$ h".format(otime[0]))
plt.axis(axe)
hqk = plt.quiverkey(hqv,0.6,0.15,2e-3,r'2 W/m${^2}$',labelpos='E',coordinates='figure')
fig.set_tight_layout(True)
if dosavefig:
    figname = 'luckyto_bclin_nrj_it{0:04}'.format(0)+pictype
    plt.savefig(path_fig+figname,bbox_inches='tight',magnification='auto',dpi=150)

#for it in range(1,nt):
for it in range(100,101):
    Fsx[indoz,indox] = (coef.squeeze()*(Fx[it,...].ravel()[elem.squeeze()])).sum(axis=-1)
    Fsz[indoz,indox] = (coef.squeeze()*(Fz[it,...].ravel()[elem.squeeze()])).sum(axis=-1)
    #hpc.set_array(nrj[it,:-1,:-1].ravel())
    hpc.set_array(nrj[it,...].ravel())
    hqv.set_UVC(Fsx,Fsz)
    plt.title(r"LUCKYTO: bclin. nrj. density at $t={}$ h".format(otime[it]))
    if dosavefig:
        figname = 'luckyto_bclin_nrj_it{0:04}'.format(it)+pictype
        plt.savefig(path_fig+figname,bbox_inches='tight',magnification='auto',dpi=150)

