# coding: utf-8

# # Calc_ModalNRJ_subdom.py (converted from jupyter notebook)
# 
# investigate modal NRJ content and transfert terms, globally and per subdomain
# 
# Conversion terms (free surface, cf Kelly 2016):
# $$ C_{mn} = u_mp_n\cdot\int\phi_m\nabla\phi_n - u_np_m\cdot\int\phi_n\nabla\phi_m $$
# 
# Adapted from a previous version (Diag_ModalNRJ_subdom). Now this notebook do all the calculation and store the results in a netCDF and csv files
# 
# See Plot_ModalNRJ_subdom.ipynb for the plots, and other notebooks as well
from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors

import numpy as np
import sys, os
from netCDF4 import Dataset#, MFDataset
from datetime import datetime
import scipy.signal as sig
from scipy.ndimage import gaussian_filter
#import scipy.interpolate as itp
from PIL import Image, ImageDraw
import json
import pandas as pd

KRYPTON = "/data0/project/vortex/lahaye/"
RUCHBA = KRYPTON+"local_ruchba/"
#scratch = os.getenv('SCRATCHDIR')+"/"
#store = os.getenv('STOREDIR')+"/"
#work = os.getenv('WORKDIR')+"/"

# informations

simul = "luckym2"
simdir = simul.upper()+"/"

data_bcl = KRYPTON+'DIAG/NRJ_fluxes/{0}_bclNRJ_M2.?.nc'.format(simul)
data_btr = KRYPTON+'DIAG/NRJ_fluxes/{0}_btrnrj.?.nc'.format(simul)
data_mod = KRYPTON+'{0}_modemap/{0}_modemap_20.nc'.format(simul)
data_nrj = KRYPTON+'{0}_modemap/{0}_modenrj_20.nc'.format(simul)

grid_file = KRYPTON+"/lucky_corgrd.nc"
doms_file = "../NRJ_flux_diag/subdomains_lucky.json"
dirpic = 'pictures/modal_conv_scatter/'
dosavefig = True 
dostore = True  # store results in a netcdf file and in a csv file

filout = KRYPTON+'{0}_modemap/{0}_mode_scatdiag_long.nc'.format(simul)
doappend = False
csvout = "./{}_NRJ_diags_long.csv".format(simul)
data_Fa14 = KRYPTON+"Tide_Conv/Falahat_etal_2014_ModalConvM2.nc"

with open(doms_file, "r") as fp:
    mydoms = json.load(fp)
      
# unfold subdomains
doms, nams = [], []
for key,val in mydoms.items():
    if key == "ridges":
        for ido,dom in enumerate(val):
            doms.append(dom)
            nams.append(key.rstrip("s")+str(ido+1))
    else:
        doms.append(val)
        nams.append(key)

rho0 = 1025

#####  ---  load stuff and declare useful functions
imod = 0 #slice(1,10)

nc = Dataset(data_nrj,'r')
times = nc.variables['time'][:]
otime = nc.variables['ocean_time'][:]
modes = nc.variables['mode'][:]
xi = nc.variables['xi_rho'][:]
eta = nc.variables['eta_rho'][:]
lon = nc.variables['lon_rho'][:]
lat = nc.variables['lat_rho'][:]
nc.close()

indt, = np.where((times>=0)&(times<=720)) #np.arange(900, 1020)
nt, it0 = len(indt), indt[0]
otime = otime[indt]

# load topo
ncgrd = Dataset(grid_file,'r')
topo = ncgrd.variables['h'][:,xi][eta,:]
dx = np.mean(1./ncgrd.variables['pm'][:,xi][eta,:])
dy = np.mean(1./ncgrd.variables['pn'][:,xi][eta,:])
dhdx = (np.gradient(ncgrd.variables['h'][:], axis=1)/dx)[:,xi][eta,:]
dhdy = (np.gradient(ncgrd.variables['h'][:], axis=0)/dy)[:,xi][eta,:]
ncgrd.close()
subsamp_step = np.diff(xi[:2])[0]
dx *= np.diff(xi[:2])[0]
dy *= np.diff(eta[:2])[0]

mast = topo<100
nmod = len(modes)
ny, nx = topo.shape

##### -- local routines (need lon, lat before for convienence)
def coord_to_pix(pos, lon=lon, lat=lat):
    return np.unravel_index(((lon-pos[0])**2+(lat-pos[1])**2).argmin(), lon.shape)[::-1]

def poly_to_mask(poly,shape):
    img = Image.new('1',shape)
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    return np.array(img)

def polycoord_tomask(coord,lon,lat):
    poly = [coord_to_pix(item, lon, lat) for item in coord]
    return poly_to_mask(poly,lon.shape)

def get_domean(field, dom, lon=lon, lat=lat, masksup=(~mast)):
    mask = polycoord_tomask(dom, lon, lat)
    if masksup is not None: mask *= masksup
    return np.nanmean(field[...,mask], axis=(-1))

#####  ---  declare pandas dataframe, initializing with a few fields
index = pd.MultiIndex.from_product([['Etot','KE','PE'],['full']+nams], names=['field', 'domain'])
datfra = pd.DataFrame(index=modes, columns=index)

#####  ---  getting netCDF file ready for storing values
if dostore and not (os.path.isfile(filout) and doappend):
    ncw = Dataset(filout, "w")
    ncw.createDimension("time", nt)
    ncw.createDimension("xi_rho", nx)
    ncw.createDimension("eta_rho", ny)
    ncw.createDimension("mode", len(modes))
    ncwar = ncw.createVariable("time", "i", ("time",))
    ncw.long_name = "time indice (output time step)"
    ncwar[:] = times[indt]
    ncwar = ncw.createVariable("xi_rho", "i", ("xi_rho",))
    ncwar.long_name = "xi indices in simulation grid"
    ncwar[:] = xi
    ncwar = ncw.createVariable("eta_rho", "i", ("eta_rho",))
    ncwar.long_name = "eta indices in simulation grid"
    ncwar[:] = eta
    ncw.createVariable("ocean_time", "f", ("time",))[:] = otime
    ncw.createVariable('lon_rho', 'f', ('eta_rho','xi_rho'))[:] = lon    
    ncw.createVariable('lat_rho', 'f', ('eta_rho','xi_rho'))[:] = lat
    ncw.generated_on = datetime.today().isoformat()
    ncw.generating_script = sys.argv[0]
    ncw.t_beg = times[indt[0]]
    ncw.t_end = times[indt[-1]]
    ncw.from_files = [data_bcl, data_btr, data_mod, data_nrj, grid_file, doms_file]
    ncw.simul_name = simul
    ncw.dxdy_subsamp = (dx, dy)
    ncw.subsamp_step = subsamp_step
    ncw.close()
times = otime
del otime

print("loaded basic stuff, declared all variables and functions, created netCDF file")
sys.stdout.flush()
#### just visualizing modal NRJ 
# WARNING barotropic PE NRJ is probably wrong

nc = Dataset(data_nrj,'r')
kemod = np.zeros((len(modes),ny,nx))
ektser = np.zeros((len(indt),len(modes)))
pemod = np.zeros((len(modes),ny,nx))
eptser = np.zeros((len(indt),len(modes)))
for it in indt:
    prov = nc.variables['ek_lf'][it,...]
    ektser[it-it0,:] = np.nanmean(prov, axis=(-1,-2))
    kemod += prov/nt
    prov = nc.variables['ep_lf'][it,...]
    eptser[it-it0,:] = np.nanmean(prov, axis=(-1,-2))
    pemod += prov/nt
etmod = kemod+pemod
nc.close()

fig, axs = plt.subplots(4, (nmod+1)//4, sharex=True, figsize=(2*(nmod+1)//4, 4.5))
for ia,ax in enumerate(axs.ravel()):
    ax.plot(times, ektser[:,ia], times, eptser[:,ia])
    ax.grid(True)

fig, axs = plt.subplots(4, (nmod+1)//4, sharex=True, sharey=True, figsize=(3*(nmod+1)//4, 4.5*3))
for ia,ax in enumerate(axs.ravel()):
    vmax = 10 if ia==0 else 2
    hpc = ax.pcolormesh(np.ma.masked_where(mast,etmod[ia,:,:]), norm=colors.LogNorm(vmin=1e-2, vmax=vmax), cmap="gist_ncar")
    ax.text(0, 1, "mode {}".format(ia), va='top', transform=ax.transAxes, bbox=dict(facecolor="white",pad=1))
    ax.set_aspect(1)

for ax in axs.ravel():
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    
fig.subplots_adjust(bottom=0.1, wspace=.04, hspace=.04)
cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.03])
ac = fig.colorbar(hpc, cax=cbar_ax, orientation="horizontal", extend="both")

if dosavefig:
    fig.savefig(dirpic+simul+"_modal_NRJ_map.png", magnification="auto", bbox_inches="tight", dpi=200)
print("loaded NRJ and plotted")
sys.stdout.flush()

#####  ---  store ke, pe and te in dataframe and netCDF file
for fie,val in zip(["Etot","KE","PE"],[etmod,kemod,pemod]):
    datfra[fie,'full'] = [np.nanmean(np.ma.masked_where(mast,val[ia,:,:])) for ia in range(nmod)]
    for nam,dom in zip(nams,doms):
        datfra[fie,nam] = get_domean(val, dom, lon=lon, lat=lat, masksup=~mast)
datfra.to_csv(csvout) # problem with csv, list are read as str
datfra.to_pickle(csvout.replace("csv","pkl"))

if dostore:
    ncw = Dataset(filout, "a")
    for var,val in zip(["KE","APE","TotE"], [kemod,pemod,etmod]):
        if var not in ncw.variables:
            ncwar = ncw.createVariable(var, "f", ("mode","eta_rho","xi_rho"))
            ncwar.longname = "modal energy"
            ncwar.units = "kJ/m^2"
        else:
            ncwar = ncw.variables[var]
        ncwar[:] = val
        ncwar.ind_t_avg = (indt[0], indt[-1])
    for var,val in zip(["KEtser","PEtser"], [ektser,eptser]):
        if var not in ncw.variables:
            ncwar = ncw.createVariable(var, "f", ("time","mode"))
            ncwar.longname = "domain-averaged modal energy"
            ncwar.units = "kJ/m^2"
        else:
            ncwar = ncw.variables[var]
        ncwar[:] = val
    ncw.close()
print("computed domain-wise average and stored time-averaged value in netCDF file")
sys.stdout.flush()

#####  --  bar plot of mean KE content
plt.figure()
plt.bar(modes[0:], np.nanmean(kemod[0:,...],axis=(1,2)))
if dosavefig:
    plt.savefig(dirpic+simul+"_mean_NRJ_per_mode.pdf", magnification="auto", bbox_inches="tight")

# plot flux divergence (take AVG)
doblur = True
siglur = 2
#indt = np.arange(nt-60-24*5, nt-60)

nc = Dataset(data_nrj,'r')
#divfmod = (np.gradient(nc.variables['Fx_avg'][:], axis=2)/dx + np.gradient(nc.variables['Fy_avg'][:], axis=1)/dy)*1e3
divfmod = np.zeros((len(modes),ny,nx))
fmodout = np.zeros((nt,len(modes)))
# plot time series for modes 1, 2, 10 and 19 over ridge2
imods = np.arange(nmod) #[1, 2, 3, 5, 10, 12, 15, 19]
divfser = np.zeros((nt,len(imods),len(nams)+1))
for it in indt:
    prov = (np.gradient(nc.variables['Fx_lf'][it,...], axis=-1)/dx 
           + np.gradient(nc.variables['Fy_lf'][it,...], axis=-2)/dy)*1e3 #W/m²
    divfmod += prov/nt #*rho0      
    fmodout[it-it0,:] = (np.diff(nc.variables['Fx_lf'][it,...][...,:,[0,nx-1]],axis=-1).sum(axis=(-1,-2))/dx \
            + np.diff(nc.variables['Fy_lf'][it,...][...,[0,ny-1],:],axis=-2).sum(axis=(-1,-2))/dy) \
            * 1e3 / (nx*ny)
    for ina,nam in enumerate(nams):
        divfser[it-it0,:,ina] = get_domean(prov[imods,:,:],doms[ina])
    ina += 1
    divfser[it-it0,:,ina] = np.nanmean(prov[:,~mast],axis=-1)[imods]
nc.close()
print("computed modal NRJ flux divergence")
sys.stdout.flush()

### plotting
for ina,nam in enumerate(nams+["full"]):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8,8))
    for imo,mod in enumerate(imods):
        ax = axs[int(mod>=10)+int(mod>0)]
        hpl, = ax.plot(times, divfser[:,imo,ina], label=str(mod))
        ax.axhline(divfser[:,imo,ina].mean(), linestyle="--", color=hpl.get_color(), linewidth=.8)
    for ax in axs.ravel():
        ax.set_xlim([times[0], times[-1]])
        ax.ticklabel_format(style="sci", scilimits=(-2,3), axis="y")
        ax.grid(True)
        ax.legend()
    axs[0].set_title('mode divergence, {} [mW/m^2]'.format(nam))
    if dosavefig:
        fig.savefig(dirpic+simul+"_modal_flux_div_{}.pdf".format(nam), magnification="auto", bbox_inches="tight", dpi=150)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8,8))
for imo,mod in enumerate(imods):
    ax = axs[int(mod>=10)+int(mod>0)]
    hpl, = ax.plot(times, fmodout[:,mod], label=str(mod))
    ax.axhline(fmodout[:,mod].mean(), linestyle="--", color=hpl.get_color(), linewidth=.8)
for ax in axs.ravel():
    ax.grid(True)
    ax.set_xlim([times[0], times[-1]])
    ax.ticklabel_format(style="sci", scilimits=(-2,3), axis="y")
    ax.legend()
axs[0].set_title('mode divergence, out [mW/m^2]')
if dosavefig:
    fig.savefig(dirpic+simul+"_modal_flux_div_out.pdf", magnification="auto", bbox_inches="tight", dpi=150)

### saving
if dostore:
    ncw = Dataset(filout, "a")
    var, val = "divf", divfmod
    if var not in ncw.variables:
        ncwar = ncw.createVariable(var, "f", ("mode","eta_rho","xi_rho"))
        ncwar.longname = "modal energy flux divergence"
        ncwar.units = "W/m^2"
    else:
        ncwar = ncw.variables[var]
    ncwar[:] = val
    ncwar.ind_t_avg = (indt[0], indt[-1])
    var, val = "divf_out", fmodout
    if (not doappend) and (var not in ncw.variables):
        ncwar = ncw.createVariable(var, "f", ("time", "mode"))
        ncwar.longname = "modal energy flux at domain boundaries"
        ncwar.units = "W/m^2"
    else:
        ncwar = ncw.variables[var]
    ncwar[:] = val
    var, val = "divf_{}", divfser
    for ina,nam in enumerate(nams+["full"]):
        if var.format(nam) not in ncw.variables:
            ncwar = ncw.createVariable(var.format(nam), "f", ("time", "mode"))
            ncwar.longname = "modal energy flux divergence averaged over {} domain".format(nam)
            ncwar.units = "W/m^2"
        else:
            ncwar = ncw.variables[var.format(nam)]
        ncwar[:] = val[:,:,ina]
    ncw.close()
print("plotted and stored")
sys.stdout.flush()


# compute over subdomains (will be exploited later)
divfmoy = {"full":np.nanmean(divfmod[:,~mast],axis=-1), "out":fmodout.mean(axis=0)}

for nam,dom in zip(nams,doms):
    divfmoy[nam] = get_domean(divfmod,dom)
for nam in ["full","out"]+nams:
    datfra['divF',nam] = divfmoy[nam]
datfra.to_csv(csvout) # problem with csv, list are read as str
datfra.to_pickle(csvout.replace("csv","pkl"))
    
# plot it
vamp = 10*np.nanstd(divfmod[1,...])
norm = colors.SymLogNorm(linthresh=vamp/10, linscale=1, vmin=-vamp, vmax=vamp)

fig, axs = plt.subplots(4, (nmod+1)//4, sharex=True, sharey=True, figsize=(2*(nmod+1)//4, 2*4.5))
for ia,ax in enumerate(axs.ravel()):
    toplot = np.ma.masked_where(mast,divfmod[ia,:,:])
    ax.text(0, 1, "mode {0} \nmean:{1:.1e}".format(ia,np.nanmean(toplot)), va='top', 
            transform=ax.transAxes, bbox=dict(facecolor="white",pad=1))
    if doblur:
        toplot = gaussian_filter(toplot, sigma=siglur)
    ax.pcolormesh(toplot, norm=norm, cmap="RdBu_r")
    ax.set_aspect(1)
    
fig.set_tight_layout(True)

print("sum over baroclinic modes: {:.2e}".format(np.nanmean(divfmod[1:,...])))
print("now computing scattering matrix")
sys.stdout.flush()

# compute every Cmn and take the mean value
# compute conversion term (no filtering) and lowpass filter / average
indm = range(nmod)
doblur = True
siglur = 2

# set up filter
dt = 1
filtperiod = 30 # in hours
bb, aa = sig.butter(4, 2*dt/30)
# m=0, n
nc = Dataset(data_mod, "r")
umod = nc.variables['u_amp']
vmod = nc.variables['v_amp']
pmod = nc.variables['p_amp']

zrho = nc.variables['zrb']

# ctmn is Cmn, with m along 1st indiex, n along 2nd index
ctmn = {nam:np.zeros((len(indm),len(indm))) for nam in nams+["full"]}

# get netCDF file ready
if dostore:
    ncw = Dataset(filout, "a")
    var = "Cmn"
    if var not in ncw.variables:
        ncwar = ncw.createVariable(var, "f", ("mode","mode","eta_rho","xi_rho"))
        ncwar.longname = "inter-modal energy conversion term, time-averaged"
        ncwar.expl = "m is first dim, n is second dim, m to n conversion, antisymmetric matrix"
        ncwar.units = "W/m^2"
    else:
        ncwar = ncw.variables[var]
    ncwar.ind_t_avg = (indt[0], indt[-1])
    ncwar.lowpass_filt_t = filtperiod
    ncwar.lowpass_filt_t_expl = "inverse cutoff frequency of lowpass filter [h]"
    var = "Cmn_tser"
    if var not in ncw.variables:
        ncwar = ncw.createVariable(var, "f", ("mode","mode","time"))
        ncwar.longname = "inter-modal energy conversion term, domain averaged"
        ncwar.expl = "m is first dim, n is second dim, m to n conversion, antisymmetric matrix"
        ncwar.units = "W/m^2"
    ncw.close()

for imod in indm:
    print("now doing mode",imod)
    sys.stdout.flush()
    phim = nc.variables['p_modes'][imod,0,...]
    lam = 1./nc.variables["mode_speed"][imod,...]**2
    for inod in indm[imod+1:]:
        if inod==imod: continue
        lan = 1./nc.variables["mode_speed"][inod,...]**2
        phin = nc.variables['p_modes'][inod,0,...]
        prav = phim*phin / (lam-lan) * (\
            pmod[indt,imod,...]*lam*(umod[indt,inod,...]*dhdx + vmod[indt,inod,...]*dhdy) \
            + pmod[indt,inod,...]*lan*(umod[indt,imod,...]*dhdx + vmod[indt,imod,...]*dhdy) )
        indy,indx = np.where(np.isfinite(prav[0,...]))
        prov = np.full((ny,nx),np.nan)
        prav = sig.filtfilt(bb, aa, prav[:,indy,indx], method="gust", axis=0)*rho0
        prov[indy,indx] = prav.mean(axis=0)
        if dostore:
            ncw = Dataset(filout, "a")
            ncw.variables["Cmn"][imod,inod,:,:] = prov
            ncw.variables["Cmn_tser"][imod,inod,:] = np.nanmean(prav[:,~mast[indy,indx]], axis=-1)
            ncw.close()
        for dom,nam in zip(doms,nams):
            ctmn[nam][imod,inod] = get_domean(prov,dom)
        ctmn["full"][imod,inod] = np.nanmean(prov)
nc.close()

# compute residual dissipation (diff between barotropic conversion and flux divergence)
# per subdomains
for nam in ["full"]+nams:
    datfra["Cmn",nam] = pd.Series([ctmn[nam][imod,imod+1:] for imod in range(nmod)])
    # dissipation per mode (tot is with barotropic conv only, res includes scattering)
    prov = ctmn[nam] - ctmn[nam].T
    datfra["Diss_res",nam] = prov.sum(axis=0) - datfra["divF",nam]
    datfra["Diss_tot",nam] = prov[0,:] - datfra["divF",nam]
datfra.to_csv(csvout) # problem with csv, list are read as str
datfra.to_pickle(csvout.replace("csv","pkl"))
    
# compute residual dissipation (diff between barotropic conversion and flux divergence)
# maps, and store

dissmod = np.full((nmod-1,ny,nx),np.nan) # btrop conv minus divF (not for btrop)
dissres = np.full((nmod,ny,nx),np.nan) # all scattering minus divF 
czon, czbc = np.full((nmod-1,ny,nx),np.nan), np.full((nmod-1,ny,nx),np.nan) # not for mode 0
if dostore:
    ncw = Dataset(filout, "a")
    varnams = ["Cbcl", "DissTot", "DissRes"]
    longnams = ["baroclinic scattering (sum Cmn over n>0)", "total dissipation (w.r.t btrop conversion)", 
                "residual dissipation (minus scattering)"]
    for var,nam in zip(varnams,longnams):
        if var not in ncw.variables:
            ncwar = ncw.createVariable(var, "f", ("mode","eta_rho","xi_rho"))
            ncwar.longname = nam
            ncwar.units = "W/m^2"
        else:
            ncwar = ncw.variables[var]
        ncwar.ind_t_avg = (indt[0], indt[-1])
    ncw.close()

ncw = Dataset(filout, "r")
for imod in range(1,nmod):
    czon[imod-1,:,:] = ncw.variables["Cmn"][0,imod,:,:]
for imod in range(1,nmod-1):
    czbc[imod-1,:,:] = ncw.variables["Cmn"][1:imod,imod,:,:].sum(axis=0) \
                        - ncw.variables["Cmn"][imod,imod+1:,:,:].sum(axis=0)
czbc[imod,:,:] = ncw.variables["Cmn"][1:imod+1,imod+1,:,:].sum(axis=0)
ncw.close()

dissmod = czon - divfmod[1:,:,:]
dissres[1:,:,:] = dissmod + czbc
dissres[0,:,:] = -czon.sum(axis=0)-divfmod[0,:,:]
if dostore:
    ncw = Dataset(filout, "a")
    ncw.variables["Cbcl"][0,:,:] = czon.sum(axis=0)
    ncw.variables["Cbcl"][1:,:,:] = czbc
    ncw.variables["DissTot"][1:,:,:] = dissmod
    ncw.variables["DissRes"][:] = dissres
    ncw.close()

### Finish the job ! store data frame into csv
datfra.to_csv(csvout) # problem with csv, list are read as str
datfra.to_pickle(csvout.replace("csv","pkl"))

