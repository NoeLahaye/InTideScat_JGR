'''
compute energy fluxes -- internal tides -- for vertical sections
adapted from comp_puv_fluxes_v4_mpi.py and comp_puv_fluxes_2D.py
butterworth filter, backward forward
No filtering (just surface/internal decomposition) 
compute pu, pv, pw, energy and barotropic terms
approx vertical integration with sigma-levels at rest (TODO fix this as soon as z_rho and 
z_w are stored in vslabs netCDF file at every time step)
NJAL June 2017
'''
from __future__ import division, print_function
from netCDF4 import Dataset
import numpy as np
import time
from scipy import signal
import R_tools as tools
import R_tools_fort as toolsF
from get_strat import get_strat
from matplotlib import pyplot as plt
plt.ion()

doverb = True

simul = 'luckyto'
path_base = '/net/krypton/data0/project/vortex/lahaye/luckyto_vslab_ridge/'
path_read = path_base+'luckyto_vslab_ridge.nc'
path_grd = '/net/ruchba/local/tmp/2/lahaye/prep_LUCKYTO/lucky_grd.nc'
path_strat = '/net/ruchba/local/tmp/2/lahaye/prep_LUCKYTO/strat_profile.nc'

dt = 1.0
#freq = 1./12.42 # 
#fwdt = 1.2      # relative width of pass band
flow = 1./40.
forder = 4      # filter order
methpad="pad"       # pad, gust (for scipy>=0.16.0)

outfile = path_base+'luckyto_nrj_fluxes.nc'

##### ---   End of user-defined parameter section   --- #####
rho0 = 1025.  # reference density, kg/m^3
grav = 9.81
p0 = 1013e2

ncr = Dataset(path_read,'r')
nt = ncr.dimensions['time'].__len__()
nl = ncr.dimensions['lslab'].__len__() 
nz = ncr.dimensions['s_rho'].__len__()

# prepare file and store (add attributes and co.)
pathwrite = outfile
if doverb:
    print('creating file',pathwrite,'for storing results')
    tmes, tmeb = time.clock(), time.time()
ncw = Dataset(pathwrite,'w',format='NETCDF4_CLASSIC')
#ncw.bp_freq_filt = freq
#ncw.bp_freq_width = fwdt
#ncw.bp_filt_type = bpfiltype
#ncw.bp_filt_order = forder
ncw.lp_freq_filt = flow
ncw.lp_filt_type = "butterworth"
ncw.lp_filt_order = forder
ncw.createDimension('time',nt)
ncw.createDimension('lslab',nl)
ncw.createDimension('s_rho',nz)
for vname in ('ocean_time','lslab','z_rho'):
    ncwar = ncw.createVariable(vname,'f',ncr[vname].dimensions)
    for attr in ncr.variables[vname].ncattrs():
        ncwar.setncattr(attr,ncr.variables[vname].getncattr(attr))
    ncwar[:] = ncr[vname][:]
# baroclinic fluxes
ncwar = ncw.createVariable('pu_bc','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"p'u': baroclinic along-slab energy flux")
ncwar.setncattr('units','kW/m^2')
ncwar = ncw.createVariable('pv_bc','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"p'v': baroclinic cross-slab energy flux")
ncwar.setncattr('units','kW/m^2')
ncwar = ncw.createVariable('pw','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"p'w': baroclinic cross-slab energy flux")
ncwar.setncattr('units','kW/m^2')
ncwar = ncw.createVariable('pubc_lf','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"low-pass filtered p'u'")
ncwar.setncattr('units','kW/m^2')
ncwar = ncw.createVariable('pvbc_lf','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"low-pass filtered p'v'")
ncwar.setncattr('units','kW/m^2')
ncwar = ncw.createVariable('pw_lf','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"low-pass filtered p'w'")
ncwar.setncattr('units','kW/m^2')
ncwar = ncw.createVariable('pulf_int','f',('time','lslab'))
ncwar.setncattr('long_name',"<p'u'>_lf vertical integral of pubc_lf")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pvlf_int','f',('time','lslab'))
ncwar.setncattr('long_name',"<p'v'>_lf vertical integral of pvbc_lf")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pwlf_int','f',('time','lslab'))
ncwar.setncattr('long_name',"<p'w'>_lf vertical integral of pw_lf")
ncwar.setncattr('units','kW/m')
# baroclinic energy
ncwar = ncw.createVariable('ek','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"ek: baroclinic kinetic energy density")
ncwar.setncattr('units','kJ/m^3')
ncwar = ncw.createVariable('ep','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"ep: baroclinic potential energy density")
ncwar.setncattr('units','kJ/m^3')
ncwar = ncw.createVariable('ek_lf','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"low-pass filtered baroclinic ke")
ncwar.setncattr('units','kJ/m^3')
ncwar = ncw.createVariable('ep_lf','f',('time','s_rho','lslab'))
ncwar.setncattr('long_name',"low-pass filtered baroclinic pe")
ncwar.setncattr('units','kJ/m^3')
ncwar = ncw.createVariable('eklf_int','f',('time','lslab'))
ncwar.setncattr('long_name',"<ek>_lf: vertical-integral of ek_lf")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('eplf_int','f',('time','lslab'))
ncwar.setncattr('long_name',"<ep>_lf: vertical-integral of ep_lf")
ncwar.setncattr('units','kJ/m^2')
# barotropic fluxes
ncwar = ncw.createVariable('pu_bt','f',('time','lslab'))
ncwar.setncattr('long_name','barotropic along-slab energy flux')
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pv_bt','f',('time','lslab'))
ncwar.setncattr('long_name','barotropic cross-slab energy flux')
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pubt_lf','f',('time','lslab'))
ncwar.setncattr('long_name','low-pass filtered btrop along-slab nrj flux')
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pvbt_lf','f',('time','lslab'))
ncwar.setncattr('long_name','low-pass filtered btrop cross-slab nrj flux')
ncwar.setncattr('units','kW/m')
# barotropic energy
ncwar = ncw.createVariable('ek_bt','f',('time','lslab'))
ncwar.setncattr('long_name','barotropic kinetic energy surf. density')
ncwar.setncattr('units','kJ/m^2')
#ncwar = ncw.createVariable('ep_bt','f',('time','lslab'))
#ncwar.setncattr('long_name','barotropic pot. energy surf. density')
#ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('ekbt_lf','f',('time','lslab'))
ncwar.setncattr('long_name','low-pass filtered btrop. kin. energy surf. density')
ncwar.setncattr('units','kJ/m^2')
#ncwar = ncw.createVariable('epbt_lf','f',('time','lslab'))
#ncwar.setncattr('long_name','low-pass filtered btrop. pot. energy surf. density')
#ncwar.setncattr('units','kJ/m^2')
# conversion terms
ncwar = ncw.createVariable('Ct','f',('time','lslab'))
ncwar.setncattr('long_name','topographic conversion term')
ncwar.setncattr('units','W')
ncwar = ncw.createVariable('Ct_lf','f',('time','lslab'))
ncwar.setncattr('long_name','low-pass filtered topo. conv. term')
ncwar.setncattr('units','W')
#ncwar = ncw.createVariable('Cs1','f',('time','lslab'))
#ncwar.setncattr('long_name','surface conversion term (numerical eta_t)')
#ncwar.setncattr('units','W')
#ncwar = ncw.createVariable('Cs1_lf','f',('time','lslab'))
#ncwar.setncattr('long_name','low-pass filtered surf. conv. term Cs1')
#ncwar.setncattr('units','W')
#ncwar = ncw.createVariable('Cs2','f',('time','lslab'))
#ncwar.setncattr('long_name','surface conversion term (numerical eta_t)')
#ncwar.setncattr('units','W')
#ncwar = ncw.createVariable('Cs2_lf','f',('time','lslab'))
#ncwar.setncattr('long_name','low-pass filtered surf. conv. term Cs2')
#ncwar.setncattr('units','W')
if doverb:
    print('file',pathwrite,'created ; took',time.clock()-tmes,time.time()-tmeb)

# prepare filter polynomials
bl, al = signal.butter(forder,flow*2*dt,btype='low')
_, p, _ = signal.tf2zpk(bl,al);     # see scipy.signal.filtfilt doc
irlen_lp = int(np.ceil(np.log(1e-9) / np.log(np.max(np.abs(p)))))
#
######  ---  start computing  ---  #####
ncvar = ncr.variables
if doverb:
    tmes, tmeb = time.clock(), time.time()
    tmis, tmib = time.clock(), time.time()
# get rest-state stratification
zr = ncvar['z_rho'][0:1,...]
zw = ncvar['z_w'][0:1,...]
dzr = np.diff(zw,axis=-1) # add empty axis for time (to be del)
#temp, salt = get_strat(path_strat,zr)
#buoybase = toolsF.get_buoy(temp,salt,zr,zw,rho0)
#bvfbase = toolsF.bvf_eos(temp,salt,zr,zw,rho0)
#pbase = -np.cumsum(dzr[...,::-1]*buoybase[...,::-1]*rho0,axis=-1)[...,::-1]
topo = -ncvar['z_w'][0,:,0]
#pbabar = np.sum(dzr*pbase,axis=-1)/topo[None,:]
#del temp; del salt; #del zr
zeta = np.zeros(nl)    # TODO change this when full z_r, z_w are stored in vslab.nc
#ncz = Dataset('/net/krypton/data0/project/vortex/lahaye/luckyto_tseries/luckyto_tseries_2Dvars.nc')      # Warning this will be removed
### baroclinic fields
uu = ncvar['ualong'][:]
vv = ncvar['ucross'][:]
pp = ncvar['pres'][:] + ncvar['pbar'][:][...,None]; 
pmoy = np.mean(pp,axis=0); pp = pp - pmoy
pbar = np.sum(dzr*pp,axis=-1)/topo[None,:]
pp = (pp - pbar[...,None])/1e3
prov = pp*uu            # along-slab bclin. energy flux
ncw['pu_bc'][:] = prov.swapaxes(1,2)
prov = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
ncw['pubc_lf'][:] = prov.swapaxes(1,2)
ncw['pulf_int'][:] = np.sum(prov*dzr,axis=-1)
prov = pp*vv            # cross-slab bclin. energy flux
ncw['pv_bc'][:] = prov.swapaxes(1,2)
prov = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
ncw['pvbc_lf'][:] = prov.swapaxes(1,2)
ncw['pvlf_int'][:] = np.sum(prov*dzr,axis=-1)
prov = (uu**2+vv**2)/2. # bclin. energy density
ncw['ek'][:] = prov.swapaxes(1,2)
prov = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
ncw['ek_lf'][:] = prov.swapaxes(1,2)
ncw['eklf_int'][:] = np.sum(prov*dzr,axis=-1)
del uu, vv
ww = ncvar['w'][:]
prov = pp*ww            # bclin vertical energy flux
ncw['pw'][:] = prov.swapaxes(1,2)
prov = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
ncw['pw_lf'][:] = prov.swapaxes(1,2)
ncw['pwlf_int'][:] = np.sum(prov*dzr,axis=-1)
pbot = pp[:,:,0]*1e3
wsurf = ww[:,:,-1]
del pp, ww
buoy = ncvar['buoy'][:]
buoybase = np.mean(buoy,axis=0)
buoy = buoy-buoybase; del buoybase
bvfmoy = tools.w2rho(np.mean(ncvar['bvf'][:],axis=0)[None,...])
prov = buoy**2/bvfmoy/2.            # bclin pot. energy density
del buoy; del bvfmoy
ncw['ep'][:] = prov.swapaxes(1,2)
prov = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
ncw['ep_lf'][:] = prov.swapaxes(1,2)
ncw['eplf_int'][:] = np.sum(dzr*prov,axis=-1)
if doverb:
    print('done with baroclinic fluxes and nrj:',time.clock()-tmis,time.time()-tmib)
    tmis, tmib = time.clock(), time.time()
### barotropic fields
ubar = ncvar['ualbar'][:]
vbar = ncvar['ucrbar'][:]
#pbar = ncvar['pbar'][:] + grav*rho0*(zeta-topo[None,:])/2. + p0
prov = 0.5*(ubar**2 + vbar**2)*topo
ncw['ek_bt'][:] = prov
ncw['ekbt_lf'][:] = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
#ncw['ep_bt'][:] = zeta**2*grav/2.
#ncw['epbt_lf'][:] = signal.filtfilt(bl,al,zeta**2*grav/2.,axis=0,method=methpad,irlen=irlen_lp)
prov = pbar*ubar*topo*1e-3
ncw['pu_bt'][:] = prov
ncw['pubt_lf'][:] = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
prov = pbar*vbar*topo*1e-3
ncw['pv_bt'][:] = prov
ncw['pvbt_lf'][:] = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
if doverb:
    print('done with barotropic fluxes and nrj:',time.clock()-tmis,time.time()-tmib)
    tmis, tmib = time.clock(), time.time()
### transfert terms
prov = -pbot*(ncvar['dhalong'][:][None,:]*ubar + ncvar['dhcross'][:][None,:]*vbar)
ncw['Ct'][:] = prov
ncw['Ct_lf'][:] = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
del pbot
#psurf = grav*zeta - pbar
#prov = np.diff(zeta,axis=0)/dt
#prov = -0.5*(prov[:-1,...] + prov[1:,...])*psurf[1:-1,...]
#ncw['Cs1'][1:-1,:] = prov
#ncw['Cs1_lf'][1:-1,:] = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
#prov = -wsurf*psurf
#ncw['Cs2'][:] = prov
#ncw['Cs2_lf'][:] = signal.filtfilt(bl,al,prov,axis=0,method=methpad,irlen=irlen_lp)
if doverb:
    print('end of computation:',time.clock()-tmes,time.time()-tmeb)

ncr.close()
ncw.close()

