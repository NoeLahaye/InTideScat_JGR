''' plot_bclin_fluxes.py 
plot baroclinic energy fluxes (raw or low-pass), from netCDF files (puv_fluxes, or bclin_fluxes)
use MFDataset to open chunked files as a single dataset
read only in the "baroclininc" netCDF file
will plot AVG as well
There are a few stuff ad hoc, as far as colorbars are concerned mainly (symlognorm...)
adapted from plot_fluxes_lf.py // LOPS version
NJAL Apr 2018 '''
from __future__ import print_function, division
dosavefig = True
import matplotlib as mpl
if dosavefig: 
    mpl.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
from mpl_toolkits.basemap import Basemap
import numpy as np
import time, os
#import scipy.integrate as itg
from netCDF4 import Dataset, MFDataset
import argparse
from diff_tools import diff_2d_ongrid

doverb = True
_simpath = "LUCKYM2"
ideb = 0           # WARNING offset (see e.g. in picture filenames)
_ires = 0          # when to start in range(ideb, ifin) (warning -1, cf. for loop)

parser = argparse.ArgumentParser()
parser.add_argument("--simpath",help="directory name of simulation",default=_simpath)
parser.add_argument("--filesuf", help="suffix for filename",default="puv_fluxes")
parser.add_argument("--field",help="field to plot (bcFlux or divFluxBC)",default="bcFlux")
parser.add_argument("-lf","--filt",help="do plot low-pass filtered",action="store_true")
parser.add_argument("--scale",help="scale for colorbar (lin/linear (default) or log)",default="linear")
parser.add_argument('--imin','-imin',help='first iteration',default=_ires)
parser.add_argument('--imax','-imax',help='last iteration')

args = parser.parse_args()

whatoplot = args.field
if whatoplot not in ["bcFlux", "divFluxBC"]:
    raise ValueError('wrong value for "field"')
if args.filt:
    whatokind = "_lf"
    ist = 12     # subsampling step if lf is on. 
else:
    whatokind = ""
    ist = 1
if args.scale == "log":
    lascale = "log"
else:
    lascale = "linear"
    
simpath = args.simpath
simname = simpath.lower()
filesuf = args.filesuf
ires = args.imin

fs       = 12
proj     = 'lcc'
res      = 'i'
stride   = 5
Lx,Ly    = 1450e3,1450e3 # extend in km
zlevs = [1000,2000,3500]
zlsty = ['-','-','--']
topocol='gray'  # isobath. contour color
grdcol='gray'   # grid color
vm = 4.
stq = 50                # step for quiver
qcol = 'darkred'       # color of arrows for nrj fluxes
if whatoplot == "bcFlux":
    if lascale == "linear":
        cmap = plt.get_cmap('gist_earth_r')  #get_colormap('WhiteBlueGreenYellowRed.ncmap')
    elif lascale == "log":
        cmap = plt.get_cmap('gist_stern_r')
    longwhat = r"$|\int \vec{F}_{bc}dz|$"
elif whatoplot == 'divFluxBC':
    cmap = 'seismic'
    longwhat = r"$\vec{\nabla}\cdot\int \vec{F}_{bc}dz$"
else:
    raise ValueError('Unrecognized whatoplot')

if simname in ['LUCKY','LUCKYT']:
    dt = 1.5
elif simname in ['LUCKYK1']:
    dt = 2.
else:
    dt = 1

path_base = '/data0/project/vortex/lahaye/DIAG/NRJ_fluxes/'
path_fig = '/data0/project/vortex/lahaye/pictures/{}/'.format(simpath)
if lascale == "linear":
    dirfig = whatoplot+whatokind+"/"
elif lascale == "log": 
    dirfig = whatoplot+whatokind+"_log/"
fil_flux = path_base+simname+'_'+filesuf+'.?.nc'
fil_grd = '/data0/project/vortex/lahaye/lucky_corgrd.nc'
pictyp = '.png'
if whatokind == "_lf":
    lpass = "low-pass "
else:
    lpass = ""
figtit = simpath+": "+lpass+longwhat
if not os.path.isdir(path_fig+dirfig):
    os.mkdir(path_fig+dirfig)

print("will do",simname,whatoplot+whatokind,"with",lascale,"color scale")
#####  ---  End of user-defined parameter section  ---  #####

if doverb:
    tmes, tmeb = time.clock(), time.time()

ncbc = MFDataset(fil_flux,aggdim='eta_rho')
if args.imax is None:
    ifin = ncbc.dimensions['time'].size
else:
    ifin = args.imax
times = np.arange(ideb,ifin+1)*dt   # WARNING scrum_time hard-coded
ist = int(ist/dt)
indx, indy = ncbc.variables['xi_rho'][:],ncbc.variables['eta_rho'][:]
st = indx[1]-indx[0]
if (indy[1]-indy[0]) != st:
    raise ValueError('subsampling step non isotropic ; cannot handle this')
slix = slice(indx[0],indx[-1]+1,st)
sliy = slice(indy[0],indy[-1]+1,st)
nc = Dataset(fil_grd,'r')
ncgvar = nc.variables
lon = ncgvar['lon_rho'][sliy,slix]
lat = ncgvar['lat_rho'][sliy,slix]
topo = ncgvar['h'][sliy,slix]
pm = ncgvar['pm'][:][sliy,slix]/st
pn = ncgvar['pn'][:][sliy,slix]/st
nc.close()
    
# initialize: use AVG
Fx = np.ma.masked_invalid(ncbc.variables['puint_avg'][:])
Fy = np.ma.masked_invalid(ncbc.variables['pvint_avg'][:])
Fnorm = np.sqrt(Fx**2 +Fy**2)
if whatoplot == "bcFlux":
    toplot = Fnorm
    clims = [0, 4*np.nanstd(toplot)]
    bextend = "max"
    try:
        units = ncbc.variables['puint_avg'].units
    except:
        units = "kW/m"
elif whatoplot == "divFluxBC":
    #toplot = np.gradient(Fx,itg.cumtrapz(1./pm, axis=1 , initial=0), axis=1) \
             #+ np.gradient(Fy,itg.cumtrapz(1./pn, axis=0 , initial=0), axis=0)
    toplot = np.pad(diff_2d_ongrid(Fx, pm, axis=1), ((0,0),(1,1)), "constant")*1e3 \
                + np.pad(diff_2d_ongrid(Fy, pn, axis=0), ((1,1),(0,0)), "constant")*1e3
    #clims = np.array([-1,1]) * 2*np.nanstd(toplot)
    clims = np.array([-1,1])*0.5    # ad hoc
    bextend = "both"
    units = r"W/m$^2$"
else:
    raise ValueError('wrong whatoplot or code error')
if lascale == "linear":
    lanorm = mpl.colors.Normalize(*clims)
elif lascale == "log":
    if whatoplot == "divFluxBC":
        lanorm = mpl.colors.SymLogNorm(linthresh=1e-2, linscale=0.5, vmin=clims[0] \
                                    , vmax=clims[1])
        tickval = np.array([0.01,0.1,0.25,0.5,1])
    elif whatoplot == "bcFlux":
        lanorm = mpl.colors.LogNorm(vmin=clims[1]/1e2)

if not dosavefig: plt.ion()
fig = plt.figure(figsize=(9,8))
ax = plt.subplot(111)
bm = Basemap(projection=proj,resolution=res,lon_0=lon.mean(),
        lat_0=lat.mean(),width=Lx,height=Ly)
xx, yy = bm(lon, lat)
bm.drawcoastlines(color='gray')
bm.fillcontinents(color='gray')
bm.drawparallels(np.arange(-60,70,stride),labels=[1,0,0,0],linewidth=0.8,\
                fontsize=fs,color=grdcol)
bm.drawmeridians(np.arange(-100,100,stride),labels=[0,0,0,1],linewidth=0.8,\
                fontsize=fs,color=grdcol)
hct = bm.contour(xx,yy,topo,levels=zlevs,colors=topocol,lw=.5,linestyles=zlsty,alpha=.6)
#if whatoplot = "bcFlux":
hpc = bm.pcolormesh(xx, yy, toplot, norm=lanorm, cmap=cmap, rasterized=True)     # just for initialization purpose
hcb = bm.colorbar(hpc, extend=bextend)
#if lascale == "log" and whatoplot == "divFluxBC":
    #hcb.set_ticks(np.sort(np.r[0,tickval,-tickval]))
hcb.set_label(units)
if whatoplot == "bcFlux":
    hpq = bm.quiver(xx[stq-1::stq,stq-1::stq], yy[stq-1::stq,stq-1::stq] \
                , Fx[stq-1::stq,stq-1::stq]/Fnorm[stq-1::stq,stq-1::stq] \
                , Fy[stq-1::stq,stq-1::stq]/Fnorm[stq-1::stq,stq-1::stq] \
                , pivot='mid', color=qcol)
for item in hct.collections:
    item.set_rasterized(True)
ax.set_title(simpath+": avg. "+longwhat, fontsize=14)
if dosavefig and ires==0:
    plt.savefig(path_fig+'{0}_{1}_avg_{2}.pdf'.format(simname,whatoplot,lascale[:3]) \
                , magnification="auto", dpi=150, bbox_inches='tight')
    
if doverb:
    print('initialization took',time.clock()-tmes,time.time()-tmeb)
    tmes, tmeb = time.clock(), time.time()

for it in range(ideb+ires,ifin,ist):
    Fx = np.ma.masked_invalid(ncbc.variables['puint'+whatokind][:,:,it])
    Fy = np.ma.masked_invalid(ncbc.variables['pvint'+whatokind][:,:,it])
    Fnorm = np.sqrt(Fx**2 +Fy**2)
    if whatoplot == "bcFlux":
        toplot = Fnorm
    elif whatoplot == "divFluxBC":
        #toplot = np.gradient(Fx,itg.cumtrapz(1./pm, axis=1 , initial=0), axis=1) \
                 #+ np.gradient(Fy,itg.cumtrapz(1./pn, axis=0 , initial=0), axis=0)
        toplot = np.pad(diff_2d_ongrid(Fx, pm, axis=1), ((0,0),(1,1)), "constant")*1e3 \
                + np.pad(diff_2d_ongrid(Fy, pn, axis=0), ((1,1),(0,0)), "constant")*1e3

##### baroclinic energy flux divergence 
    hpc.set_array(toplot[:-1,:-1].ravel())
    if whatoplot == "bcFlux":
        hpq.set_UVC(Fx[stq-1::stq,stq-1::stq]/Fnorm[stq-1::stq,stq-1::stq] \
                , Fy[stq-1::stq,stq-1::stq]/Fnorm[stq-1::stq,stq-1::stq])
    ax.set_title(figtit+" at t={:.1f}".format(times[it]))
    if dosavefig:
        plt.savefig(path_fig+dirfig+'{0}_{1}_t{2:04}'.format(simname,whatoplot+whatokind,it)+pictyp \
                    , dpi=150, bbox_inches='tight', magnification='auto')
    if doverb:
        print('it {0}, plot div bc:'.format(it),time.clock()-tmes,time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()

ncbc.close()

