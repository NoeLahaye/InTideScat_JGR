''' comp_modal_nrjflux.py
compute horizontal flux of vertically integrated energy 
and energy density ; 
store in a netCDF file
NJAL October 2017 '''
from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset
from scipy import signal
import time
from xrdataset import xrdataset

doverb = True

simul = 'luckyto'
#dirfil = '/ccc/scratch/cont003/gen7638/lahayen/{}/data/modemap/'.format(simul.upper())
dirfil = '/data0/project/vortex/lahaye/{}_modemap/'.format(simul)
dirfil = '/net/krypton'+dirfil # if not on libra
dirout = '/net/ruchba/local/tmp/2/lahaye/{}_modemap/'.format(simul)
filenames = [[dirfil+simul+'_modemap.{0}_{1}.nc'.format(jj,ii) for ii in range(8)] for jj in range(8)]
aggdims = ['eta_rho','xi_rho']
filout = simul+'_mod{:02d}_nrj.nc'

imodes = 'all'              # 'all' or numpy array

freq = 1./24. # 
forder = 4      # filter order
bpfiltype = 'butter'
methpad="gust"       # pad, gust (for scipy>=0.16.0)

grav = 9.81     # m/s^2
rho0 = 1.025    # reference density, Mg/m^3

#####   --- netCDF files    --- #####
#ncr = Dataset(dirfil+filename,'r')
ncr = xrdataset(filenames,aggdims)

def createncfile(ncin,filename,imod):
    ncw = Dataset(filename,'w')
	### copy dimensions and fix variables
    for dim in ['time','xi_rho','eta_rho']:
	    ncw.createDimension(dim,ncin.dims[dim])
	    ncw.createVariable(dim,'i2',ncin.variables[dim].dims)[:] = ncin.variables[dim][:]
    dim = 'mode'
    ncw.createDimension(dim,1)
    ncw.createVariable(dim,'i2',ncin.variables[dim].dims)[:] = ncin.variables[dim][imod]
    for var in ['lon_rho','lat_rho','topo','ocean_time']:
        ncw.createVariable(var,'f',ncin.variables[var].dims)[:] = ncin.variables[var][:]
    var = 'mode_speed'
    ncw.createVariable(var,'f',ncin.variables[var].dims)[:] = ncin.variables[var][imod,:,:]
    ### create new variables
    ncw.createVariable('Fx','f',ncin.variables['p_amp'].dims)
    ncw.longname = 'Horizontal modal flux, x-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fy','f',ncin.variables['p_amp'].dims)
    ncw.longname = 'Horizontal modal flux, y-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fx_lf','f',ncin.variables['p_amp'].dims)
    ncw.longname = 'lowpass filtered horizontal modal flux, x-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fy_lf','f',ncin.variables['p_amp'].dims)
    ncw.longname = 'lowpass filtered horizontal modal flux, y-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fx_avg','f',ncin.variables['mode_speed'].dims)
    ncw.longname = 'time-averaged horizontal modal flux, x-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fy_avg','f',ncin.variables['mode_speed'].dims)
    ncw.longname = 'time-averaged horizontal modal flux, y-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('ek','f',ncin.variables['p_amp'].dims)
    ncw.longname = 'modal kinetic energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ep','f',ncin.variables['p_amp'].dims)
    ncw.longname = 'modal potential energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ek_lf','f',ncin.variables['p_amp'].dims)
    ncw.longname = 'low-pass filtered modal kinetic energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ep_lf','f',ncin.variables['p_amp'].dims)
    ncw.longname = 'low-pass filtered modal potential energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ek_avg','f',ncin.variables['mode_speed'].dims)
    ncw.longname = 'time-averaged modal kinetic energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ep_avg','f',ncin.variables['mode_speed'].dims)
    ncw.longname = 'time-averaged modal potential energy surface density'
    ncw.units = 'kJ/m^2'
    return ncw

if imodes == 'all':
    imodes = np.arange(ncr.dims['mode'])

#### compute norm
zw = ncr.variables['zwb'][:].values
phi = ncr.variables['w_modes'][imodes,...].values
Nsqb = ncr.variables['Nsqb'].values
norm_w = np.trapz(Nsqb[None,...]*phi**2,zw[None,...],axis=1) + grav*phi[:,-1,...]**2 # modified norm
zw = np.diff(zw,axis=0)
phi = ncr.variables['p_modes'][imodes,...].values
norm_p = np.sum(zw[None,...]*phi**2,axis=1)
del zw, phi, Nsqb
if doverb:
    print('done with initialization, start computing')
    tmes, tmeb = time.clock(), time.time()

#####   --- compute fluxes  --- #####
dt = np.diff(ncr.variables['ocean_time'][:2])
bb, aa = signal.butter(forder,freq*2.*dt,btype='low')
for imod in imodes:
    ncw = createncfile(ncr,dirfil+filout.format(imod),imod)
    ncwar = ncw.variables
    indoy, indox = np.where(np.isfinite(norm_p[imod,...]))
    data = ncr.variables['p_amp'][:,imod,...]*ncr.variables['u_amp'][:,imod,...]\
                *norm_p[imod,...]#*rho0      # kW/m
    ncwar['Fx'][0,...] = data
    ncwar['Fx_avg'][0,...] = np.nanmean(data,axis=0)
    data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
    ncwar['Fx_lf'][0,...] = data
    data = ncr.variables['p_amp'][:,imod,...]*ncr.variables['v_amp'][:,imod,...]\
                *norm_p[imod,...]#*rho0  
    ncwar['Fy'][0,...] = data
    ncwar['Fy_avg'][0,...] = np.nanmean(data,axis=0)
    data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
    ncwar['Fy_lf'][0,...] = data
    data = (ncr.variables['u_amp'][:,imod,...]**2 + ncr.variables['v_amp'][:,imod,...]**2)\
            *norm_p[imod,...]/2.*rho0       # kJ/m^2
    ncwar['ek'][0,...] = data
    ncwar['ek_avg'][0,...] = np.nanmean(data,axis=0)
    data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
    ncwar['ek_lf'][0,...] = data
    data = ncr.variables['b_amp'][:,imod,...]**2./2.*norm_w[imod,...]*rho0
    ncwar['ep'][0,...] = data
    ncwar['ep_avg'][0,...] = np.nanmean(data,axis=0)
    data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
    ncwar['ep_lf'][0,...] = data
    ncw.close()
    if doverb:
        print('mode {} done ; timing:'.format(imod),time.clock()-tmes,time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()

