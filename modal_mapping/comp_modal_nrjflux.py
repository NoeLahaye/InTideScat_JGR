''' comp_modal_nrjflux.py
compute horizontal flux of vertically integrated energy 
and energy density ; 
store in a netCDF file
N.B.: pressure in modes is reduced pressure p/rho0
same as comp_modal_nrjflux but mode is first dimension instead of time
NJAL October 2017 '''
from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset
from scipy import signal
import time, os, sys
from datetime import datetime

doverb = True

nmod = 20

simulpath = "LUCKYM2"
simul = simulpath.lower()
#dirfil = '/ccc/store/cont003/gen7638/lahayen/{}/data/modemap/'.format(simulpath)
dirfil = '/data0/project/vortex/lahaye/{}_modemap/'.format(simul)
dirout = dirfil
filename = simul+'_modemap_{:d}.nc'.format(nmod)
filout = simul+'_modenrj_{:d}.nc'.format(nmod)
doappend = False

imodes = 'all' #np.arange(2)         # 'all' or numpy array

freq = 1./30. # 
forder = 4      # filter order
bpfiltype = 'butter'
methpad = "gust"       # pad, gust (for scipy>=0.16.0)

grav = 9.81     # m/s^2
rho0 = 1025.    # reference density, kg/m^3

#####   --- netCDF files    --- #####
ncr = Dataset(dirfil+filename,'r')
#ncr = xrDataset(dirfil+filename,('xi_rho','eta_rho'))
if doappend and os.path.isfile(dirfil+filout):
    ncw = Dataset(dirfil+filout,'r+')
else:
    ncw = Dataset(dirfil+filout,'w')
    ### copy dimensions and fix variables
    if imodes == 'all':
        imodes = np.arange(ncr.dimensions['mode'].size)
    for dim in ['time','xi_rho','eta_rho']:
        ncw.createDimension(dim,ncr.dimensions[dim].size)
        ncw.createVariable(dim,'i2',ncr.variables[dim].dimensions)[:] = ncr.variables[dim][:]
    dim = 'mode'
    ncw.createDimension(dim,len(imodes))
    ncw.createVariable(dim,'i2',ncr.variables[dim].dimensions) 
    for var in ['lon_rho','lat_rho','topo','ocean_time']:
        ncw.createVariable(var,'f',ncr.variables[var].dimensions)[:] = ncr.variables[var][:]
    var = 'mode_speed'
    ncw.createVariable(var,'f',ncr.variables[var].dimensions) 
    ### create new variables
    ncwar = ncw.createVariable('Fx','f',ncr.variables['p_amp'].dimensions)
    ncwar.longname = 'Horizontal modal flux, x-direction'
    ncwar.units = 'kW/m'
    ncwar = ncw.createVariable('Fy','f',ncr.variables['p_amp'].dimensions)
    ncwar.longname = 'Horizontal modal flux, y-direction'
    ncwar.units = 'kW/m'
    ncwar = ncw.createVariable('Fx_lf','f',ncr.variables['p_amp'].dimensions)
    ncwar.longname = 'lowpass filtered horizontal modal flux, x-direction'
    ncwar.units = 'kW/m'
    ncwar = ncw.createVariable('Fy_lf','f',ncr.variables['p_amp'].dimensions)
    ncwar.longname = 'lowpass filtered horizontal modal flux, y-direction'
    ncwar.units = 'kW/m'
    ncwar = ncw.createVariable('Fx_avg','f',ncr.variables['mode_speed'].dimensions)
    ncwar.longname = 'time-averaged horizontal modal flux, x-direction'
    ncwar.units = 'kW/m'
    ncwar = ncw.createVariable('Fy_avg','f',ncr.variables['mode_speed'].dimensions)
    ncwar.longname = 'time-averaged horizontal modal flux, y-direction'
    ncwar.units = 'kW/m'
    ncwar = ncw.createVariable('ek','f',ncr.variables['p_amp'].dimensions)
    ncwar.longname = 'modal kinetic energy surface density'
    ncwar.units = 'kJ/m^2'
    ncwar = ncw.createVariable('ep','f',ncr.variables['p_amp'].dimensions)
    ncwar.longname = 'modal potential energy surface density'
    ncwar.units = 'kJ/m^2'
    ncwar = ncw.createVariable('ek_lf','f',ncr.variables['p_amp'].dimensions)
    ncwar.longname = 'low-pass filtered modal kinetic energy surface density'
    ncwar.units = 'kJ/m^2'
    ncwar = ncw.createVariable('ep_lf','f',ncr.variables['p_amp'].dimensions)
    ncwar.longname = 'low-pass filtered modal potential energy surface density'
    ncwar.units = 'kJ/m^2'
    ncwar = ncw.createVariable('ek_avg','f',ncr.variables['mode_speed'].dimensions)
    ncwar.longname = 'time-averaged modal kinetic energy surface density'
    ncwar.units = 'kJ/m^2'
    ncwar = ncw.createVariable('ep_avg','f',ncr.variables['mode_speed'].dimensions)
    ncwar.longname = 'time-averaged modal potential energy surface density'
    ncwar.units = 'kJ/m^2'
    ncw.generated_on = datetime.today().isoformat()
    ncw.generating_script = sys.argv[0]
    ncw.lowf_filter_period = 1/freq
    ncw.from_files = dirfil+filename
    ncw.simul = simul
ncwar = ncw.variables

#### compute norm
zw = ncr.variables['zwb'][:]
phi = ncr.variables['w_modes'][imodes,...]
Nsqb = ncr.variables['Nsqb'][:]
norm_w = np.trapz(Nsqb[None,...]*phi**2,zw[None,...],axis=1) + grav*phi[:,-1,...]**2 # modified norm
zw = np.diff(zw,axis=0)
phi = ncr.variables['p_modes'][imodes,...]
norm_p = np.sum(zw[None,...]*phi**2,axis=1)
del zw, phi, Nsqb
if doverb:
    print('done with initialization, start computing')
    tmes, tmeb = time.clock(), time.time()

#####   --- compute fluxes  --- #####
dt = np.diff(ncr.variables['ocean_time'][:2])
bb, aa = signal.butter(forder,freq*2.*dt,btype='low')
im0 = imodes[0]
for imod in imodes:
    indoy, indox = np.where(np.isfinite(norm_p[imod-im0,...]))
    data = ncr.variables['p_amp'][:,imod,...]*ncr.variables['u_amp'][:,imod,...]\
                *norm_p[imod-im0,...]*rho0/1e3      # kW/m
    ncwar['Fx'][:,imod,...] = data
    ncwar['Fx_avg'][imod,...] = np.nanmean(data,axis=0)
    data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
    ncwar['Fx_lf'][:,imod,...] = data
    data = ncr.variables['p_amp'][:,imod,...]*ncr.variables['v_amp'][:,imod,...]\
                *norm_p[imod-im0,...]*rho0/1e3  
    ncwar['Fy'][:,imod,...] = data
    ncwar['Fy_avg'][imod,...] = np.nanmean(data,axis=0)
    data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
    ncwar['Fy_lf'][:,imod,...] = data
    data = (ncr.variables['u_amp'][:,imod,...]**2 + ncr.variables['v_amp'][:,imod,...]**2)\
            *norm_p[imod-im0,...]/2.*rho0/1e3       # kJ/m^2
    ncwar['ek'][:,imod,...] = data
    ncwar['ek_avg'][imod,...] = np.nanmean(data,axis=0)
    data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
    ncwar['ek_lf'][:,imod,...] = data
    data = ncr.variables['b_amp'][:,imod,...]**2./2.*norm_w[imod-im0,...]*rho0/1e3
    ncwar['ep'][:,imod,...] = data
    ncwar['ep_avg'][imod,...] = np.nanmean(data,axis=0)
    data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
    ncwar['ep_lf'][:,imod,...] = data
    ncwar['mode'][imod] = ncr.variables["mode"][imod]
    ncwar["mode_speed"][imod,:,:] = ncr.variables["mode_speed"][imod,:,:]
    ncw.sync()
    if doverb:
        print('mode {} done ; timing:'.format(imod),time.clock()-tmes,time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()

ncw.close()
ncr.close()

