''' comp_modal_nrjflux.py
compute horizontal flux of vertically integrated energy 
and energy density ; 
read per chunk, compute every modes at the same time per chunk and store in mode-chunked 
files (in addition to x-chunked and y-chunked: small files strategy)
from comp_modal_nrjflux_chunked.py ; NJAL April 2018 '''
from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset
from scipy import signal
import time, sys
from datetime import datetime
import itertools
#from xrdataset import xrdataset
#from xarray import open_mfdataset as xrdataset

doverb = True

simul = 'luckyto'
#dirfil = '/ccc/scratch/cont003/gen7638/lahayen/{}/data/modemap/'.format(simul.upper())
dirfil = '/data0/project/vortex/lahaye/{}_modemap/'.format(simul)
dirfil = '/net/krypton'+dirfil # if not on libra
dirout = '/net/ruchba/local/tmp/2/lahaye/{}_modemap/'.format(simul)
filenames = dirfil+simul+'_modemap.{0}_{1}.nc'
filout = simul+'_mod{0:02d}_nrj.{1}_{2}.nc'

imodes = np.arange(10) #'all'   # 'all' or numpy array

freq = 1./24. # 
forder = 4      # filter order
bpfiltype = 'butter'
methpad = "gust"       # pad, gust (for scipy>=0.16.0)

grav = 9.81     # m/s^2
rho0 = 1.025    # reference density, Mg/m^3

#####   --- netCDF files    --- #####
ncr = Dataset(filenames.format(0,0),'r')
if imodes == 'all':
    imodes = np.arange(ncr.dimensions['mode'].size)
dt = np.diff(ncr.variables['ocean_time'][:2])
bb, aa = signal.butter(forder,freq*2.*dt,btype='low')
npx, npy = ncr.npx, ncr.npy
ncr.close()
if len(sys.argv) > 1:
    ipy = [int(sys.argv[1])]
else:
    ipy = np.arange(npy)
ipx = np.arange(npx)

def createncfile(ncin,filename,imod):
    # ncin a xarray dataset
    ncw = Dataset(filename,'w')
    for atnam in ncin.ncattrs():
        ncw.setncattr(atnam, ncin.getncattr(atnam))
    ncw.setncattr('generating_script', sys.argv[0])
    ncw.setncattr('generated_on', datetime.today().isoformat())
	### copy dimensions and fix variables
    for dim in ['time','xi_rho','eta_rho']:
	    ncw.createDimension(dim,ncin.dimensions[dim].size)
	    ncw.createVariable(dim,'i2',ncin.variables[dim].dimensions)[:] = ncin.variables[dim][:]
    dim = 'mode'
    ncw.createDimension(dim,1)
    ncw.createVariable(dim,'i2',ncin.variables[dim].dimensions)[:] = ncin.variables[dim][imod]
    for var in ['lon_rho','lat_rho','topo','ocean_time']:
        ncw.createVariable(var,'f',ncin.variables[var].dimensions)[:] = ncin.variables[var][:]
    var = 'mode_speed'
    ncw.createVariable(var,'f',ncin.variables[var].dimensions)[:] = ncin.variables[var][imod,:,:]
    ### create new variables
    ncw.createVariable('Fx','f',ncin.variables['p_amp'].dimensions)
    ncw.longname = 'Horizontal modal flux, x-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fy','f',ncin.variables['p_amp'].dimensions)
    ncw.longname = 'Horizontal modal flux, y-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fx_lf','f',ncin.variables['p_amp'].dimensions)
    ncw.longname = 'lowpass filtered horizontal modal flux, x-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fy_lf','f',ncin.variables['p_amp'].dimensions)
    ncw.longname = 'lowpass filtered horizontal modal flux, y-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fx_avg','f',ncin.variables['mode_speed'].dimensions)
    ncw.longname = 'time-averaged horizontal modal flux, x-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('Fy_avg','f',ncin.variables['mode_speed'].dimensions)
    ncw.longname = 'time-averaged horizontal modal flux, y-direction'
    ncw.units = 'kW/m'
    ncw.createVariable('ek','f',ncin.variables['p_amp'].dimensions)
    ncw.longname = 'modal kinetic energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ep','f',ncin.variables['p_amp'].dimensions)
    ncw.longname = 'modal potential energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ek_lf','f',ncin.variables['p_amp'].dimensions)
    ncw.longname = 'low-pass filtered modal kinetic energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ep_lf','f',ncin.variables['p_amp'].dimensions)
    ncw.longname = 'low-pass filtered modal potential energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ek_avg','f',ncin.variables['mode_speed'].dimensions)
    ncw.longname = 'time-averaged modal kinetic energy surface density'
    ncw.units = 'kJ/m^2'
    ncw.createVariable('ep_avg','f',ncin.variables['mode_speed'].dimensions)
    ncw.longname = 'time-averaged modal potential energy surface density'
    ncw.units = 'kJ/m^2'
    return ncw

### Start loop over y-chunks, concatenating x-chunks
if doverb:
    print("starting loop over chunks")
for jy,ix in itertools.product(ipy, ipx):
    if doverb:
        tmes, tmeb = time.clock(), time.time()
    fname = filenames.format(jy, ix)
    ncr = Dataset(fname, 'r')
    #### compute norm
    zw = ncr.variables['zwb'][:]
    phi = ncr.variables['w_modes'][imodes,...]
    Nsqb = ncr.variables['Nsqb'][:]
    norm_w = np.trapz(Nsqb[None,...]*phi**2,zw[None,...],axis=1) + grav*phi[:,-1,...]**2 # modified norm
    zw = np.diff(zw,axis=0)
    phi = ncr.variables['p_modes'][imodes,...]
    norm_p = np.sum(zw[None,...]*phi**2,axis=1)
    del zw, phi, Nsqb
    
    ncw = []
    ### create netCDF files
    for imod,nmod in zip(range(len(imodes)),imodes):
        ncw.append( createncfile(ncr, dirfil+filout.format(nmod,jy,ix), nmod))
        ncw[-1].from_file = fname

    #####   --- compute fluxes  --- #####
    if doverb:
        print('netCDF file created, done initializing for chunk ; timing:', time.clock()-tmes \
                , time.time()-tmeb)
        print('  ---  start computing  ---')
        tmes, tmeb = time.clock(), time.time()
    indoy, indox = np.where(np.isfinite(norm_p[nmod,...]))
    data = ncr.variables['p_amp'][:,imodes,...]*ncr.variables['u_amp'][:,imodes,...]\
                *norm_p[imodes,...]*rho0      # kW/m
    for imod in range(len(imodes)):
        ncw[imod].variables['Fx'][:] = data[:,imod,...]
        ncw[imod].variables['Fx_avg'][:] = np.nanmean(data[:,imod,...],axis=0)
    data[...,indoy,indox] = signal.filtfilt(bb,aa,data[...,indoy,indox],axis=0,method=methpad)
    for imod in range(len(imodes)):
        ncw[imod].variables['Fx_lf'][:] = data[:,imod,...]
    data = ncr.variables['p_amp'][:,imodes,...]*ncr.variables['v_amp'][:,imodes,...]\
                *norm_p[imodes,...]*rho0  
    for imod in range(len(imodes)):
        ncw[imod].variables['Fy'][:] = data[:,imod,...]
        ncw[imod].variables['Fy_avg'][:] = np.nanmean(data[:,imod,...],axis=0)
    data[...,indoy,indox] = signal.filtfilt(bb,aa,data[...,indoy,indox],axis=0,method=methpad)
    for imod in range(len(imodes)):
        ncw[imod].variables['Fy_lf'][:] = data[:,imod,...]
    data = (ncr.variables['u_amp'][:,imodes,...]**2 + ncr.variables['v_amp'][:,imodes,...]**2)\
            *norm_p[imodes,...]/2.*rho0       # kJ/m^2
    for imod in range(len(imodes)):
        ncw[imod].variables['ek'][:] = data[:,imod,...]
        ncw[imod].variables['ek_avg'][:] = np.nanmean(data[:,imod,...],axis=0)
    data[...,indoy,indox] = signal.filtfilt(bb,aa,data[...,indoy,indox],axis=0,method=methpad)
    for imod in range(len(imodes)):
        ncw[imod].variables['ek_lf'][:] = data[:,imod,...]
    data = ncr.variables['b_amp'][:,imodes,...]**2./2.*norm_w[imodes,...]*rho0
    ncr.close()
    for imod in range(len(imodes)):
        ncw[imod].variables['ep'][:] = data[:,imod,...]
        ncw[imod].variables['ep_avg'][:] = np.nanmean(data[:,imod,...],axis=0)
    data[...,indoy,indox] = signal.filtfilt(bb,aa,data[...,indoy,indox],axis=0,method=methpad)
    for imod in range(len(imodes)):
        ncw[imod].variables['ep_lf'][:] = data[:,imod,...]
        ncw[imod].close()
    if doverb:
        print('chunk {0},{1} done ; timing:'.format(jy,ix) \
                , time.clock()-tmes, time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()
    
