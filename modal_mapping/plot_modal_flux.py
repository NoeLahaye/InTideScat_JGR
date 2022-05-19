''' plot_modal_flux.py
read modal fluxes (produced with comp_modal_nrjfluxes) in netCDF
plot snapshots, low-pass and/or time-average
WARNING: imodes is not well defined and well used (will work only if range(nmodes))
NJAL Oct 2017 '''
from __future__ import print_function, division
dosavefig = True
import matplotlib as mpl
if dosavefig: 
    mpl.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset, MFDataset
import numpy as np
import time, os
from scipy.ndimage import filters

doverb = True

simpath = "LUCKYM2s"
simul = simpath.lower()
dirfil = '/'.join([os.getenv('STOREDIR'),simpath,'/data/modemap/'])
# will create folders here to save pictures
dirpic = '/'.join([os.getenv('WORKDIR'),simpath,'pictures/modal_mapping/'])
filename = simul+'_modenrj.nc'          # netCDF file to read from

ideb = 0           # WARNING offset (see e.g. in picture filenames)
ires = 0          # when to start in range(ideb, ifin) (warning -1, cf. for loop)
ifin = 720         
istp = 1         # time step

imodes = 'all' # 'all'              # mode number to plot: 'all' or a list (range)
kinds = ['avg','lf']  #['raw','lf','avg'] # list
cmap_abs = 'gist_earth_r'
lev_h = [-4000,-3000,-2000,-1000]
lst_h = ['-','-','-','-']
col_h = ['grey','grey','k','k']

pictype = '.png'  # picture output format
fs       = 12       # fontsize
proj     = 'lcc'    # grid projection
res      = 'i'
stride   = 5
grdcol = 'k'        # grid line color
stq = 20                # step for quiver
qcol = 'darkred'       # color of arrows for nrj fluxes
qsc = 5.        # scale for quiver (au pif)
#qwdt = 1.0
blur = 3        # None or #points, bluring for quiver (not sure None renders correct)

rho0 = 1.025e3  # this won't be usefull in future versions

#####   --- End of user-defined parameter section   --- #####
#############################################################

suffix = {'raw':'', 'lf':'_lf','avg':'_avg'}

ncr = Dataset(dirfil+filename,'r')
lon = ncr.variables['lon_rho'][:]   # shape (y,x)
lat = ncr.variables['lat_rho'][:]
topo = -ncr.variables['topo'][:]
Lx = (lon[:,-1].mean() - lon[:,0].mean())*6400e3*np.cos(lat.mean()*np.pi/180.)*np.pi/180.
Ly = (lat[-1,:].mean() - lat[0,:].mean())*6400e3*np.pi/180.
if imodes == 'all':
    imodes = np.arange(ncr.dimensions['mode'].size)
modes = ncr.variables['mode'][imodes]

### Deal with repertories to store pictures
for kin in kinds:
    if not os.path.isdir(dirpic+'modflux'+suffix[kin]):
        os.makedirs(dirpic+'modflux'+suffix[kin])

times = ncr.variables['ocean_time'][:]

if doverb:
    tmes, tmeb = time.clock(), time.time()

if not dosavefig: plt.ion()
# initialize plot using average 
fig = plt.figure()
ax = plt.subplot(111)
bm = Basemap(projection=proj,resolution=res,\
            lon_0=lon.mean(),lat_0=lat.mean(),\
            width=Lx,height=Ly)
xx,yy = bm(lon,lat) 
# --- map ---                                  
bm.drawcoastlines(color='gray')
bm.fillcontinents(color='gray')
hpar = bm.drawparallels(np.arange(-60,70,stride),labels=[1,0,0,0],linewidth=0.5,fontsize=fs,color=grdcol)
bm.drawmeridians(np.arange(-100,100,stride),labels=[0,0,0,1],linewidth=0.5,fontsize=fs,color=grdcol)
# extend grid for pcolormesh (better to pass edges)
xmesh = np.r_['-1',(1.5*xx[:,0]-0.5*xx[:,1])[:,None],0.5*(xx[:,1:]+xx[:,:-1]),\
                    (1.5*xx[:,-1]-0.5*xx[:,-2])[:,None]]
ymesh = np.r_['-1',(1.5*yy[:,0]-0.5*yy[:,1])[:,None],0.5*(yy[:,1:]+yy[:,:-1]),\
                    (1.5*yy[:,-1]-0.5*yy[:,-2])[:,None]]
xmesh = np.vstack((1.5*xmesh[0,:]-0.5*xmesh[1,:],0.5*(xmesh[1:,:]+xmesh[:-1,:]),\
                    1.5*xmesh[-1,:]-0.5*xmesh[-2,:]))
ymesh = np.vstack((1.5*ymesh[0,:]-0.5*ymesh[1,:],0.5*(ymesh[1:,:]+ymesh[:-1,:]),\
                    1.5*ymesh[-1,:]-0.5*ymesh[-2,:]))
hpc = bm.pcolormesh(xmesh,ymesh,np.full(topo.shape,np.nan,dtype=np.float),cmap=cmap_abs)
hcb = bm.colorbar(hpc,extend="both")
hcb.formatter.set_powerlimits((-2,2))
# --- topo ---
hct = plt.contour(xx,yy,topo,levels=lev_h,colors=col_h,linewidths=0.7,linestyles=lst_h,latlon=True)    
ax.tick_params(labelsize=fs)
hpq = bm.quiver(xx[::stq,::stq],yy[::stq,::stq],xx[::stq,::stq],yy[::stq,::stq]\
                    ,pivot='tail',color=qcol,scale=qsc)
for item in hct.collections:
    item.set_rasterized(True)
if doverb:
    print('initialization took',time.clock()-tmes,time.time()-tmeb)
    tmes, tmeb = time.clock(), time.time()

### start with average
limsc = []
slix = slice(stq//4,xx.shape[-1],stq)
sliy = slice(stq//4,xx.shape[0],stq)
for imod in imodes:
    Fx = np.ma.masked_invalid(ncr.variables['Fx_avg'][imod,...])*rho0
    Fy = np.ma.masked_invalid(ncr.variables['Fy_avg'][imod,...])*rho0
    Fnorm = np.sqrt(Fx**2 +Fy**2)
    if modes[imod] == 0:
        limsc.append(0.75*Fnorm.max())   # clims automatic from time-average
    else:
        limsc.append(5*np.nanstd(Fnorm))
    if 'avg' in kinds:
        hpc.set_array(Fnorm.ravel())
        hpc.set_clim(vmin=0,vmax=limsc[-1])
        if not blur is None:
            Fnorm = filters.gaussian_filter(Fnorm,blur)
            Fx = filters.gaussian_filter(Fx,blur)/Fnorm
            Fy = filters.gaussian_filter(Fy,blur)/Fnorm
            Fnorm = np.log10(1+np.minimum(Fnorm/limsc[-1],np.ones(Fnorm.shape)))
        #hpq.set_UVC(Fx[::stq,::stq]/Fnorm[::stq,::stq],Fy[::stq,::stq]/Fnorm[::stq,::stq])
        hpq.set_UVC(Fx[sliy,slix]*Fnorm[sliy,slix],Fy[sliy,slix]*Fnorm[sliy,slix])
        #hpq.set_UVC(Fx[::stq,::stq]/limsc[imod],Fy[::stq,::stq]/limsc[imod])
        #hpq.set_UVC(Fx[::stq,::stq]*np.log10(1+Fnorm[::stq,::stq]),Fy[::stq,::stq]*np.log10(1+Fnorm[::stq,::stq]))
        letitre = simul.upper()+r' mode {0} NRJ flux, time-averaged (kW/m)'.format(modes[imod])
        plt.title(letitre)
        if dosavefig:
            pathpic = dirpic+'modflux'+suffix['avg']+'/'+simul\
                        +'_modflux_m{}_avg'.format(modes[imod])+pictype
            plt.savefig(pathpic,magnification='auto',bbox_inches='tight',dpi=200)
if 'avg' in kinds:
    kinds.remove('avg')
    if doverb:
        print('end dealing with time average',time.clock()-tmes,time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()

### now make lf and raw
if 'lf' in kinds or 'raw' in kinds:
    for it in range(ideb+ires,ifin,istp):
        for kin in kinds:
            for imod in imodes:
                Fx = np.ma.masked_invalid(ncr.variables['Fx'+suffix[kin]][it,imod,...])*rho0
                Fy = np.ma.masked_invalid(ncr.variables['Fy'+suffix[kin]][it,imod,...])*rho0
                Fnorm = np.sqrt(Fx**2 +Fy**2)
                hpc.set_array(Fnorm.ravel())
                hpc.set_clim(vmin=0,vmax=limsc[imod])
                if not blur is None:
                    Fnorm = filters.gaussian_filter(Fnorm,blur)
                    Fx = filters.gaussian_filter(Fx,blur)/Fnorm
                    Fy = filters.gaussian_filter(Fy,blur)/Fnorm
                    Fnorm = np.log10(1+np.minimum(Fnorm/limsc[imod],np.ones(Fnorm.shape)))
                #hpq.set_UVC(Fx[::stq,::stq]/Fnorm[::stq,::stq],Fy[::stq,::stq]/Fnorm[::stq,::stq])
                hpq.set_UVC(Fx[sliy,slix]*Fnorm[sliy,slix],Fy[sliy,slix]*Fnorm[sliy,slix])
                letitre = simul.upper()+r' mode {0} NRJ flux, {1} (kW/m) -- t={2}'\
                            .format(modes[imod],kin,times[it])
                plt.title(letitre)
                if dosavefig:
                    pathpic = dirpic+'modflux'+suffix[kin]+'/'+simul\
                        +'_modflux_m{0}_t{1:03d}'.format(modes[imod],it)+pictype
                    plt.savefig(pathpic,magnification='auto',bbox_inches='tight',dpi=200)
        if doverb:
            print('it. {0}/{1} took'.format(it,ifin),time.clock()-tmes,time.time()-tmeb)
            tmes, tmeb = time.clock(), time.time()

