''' comp_modal_nrjflux_chunked.py
compute horizontal flux of vertically integrated energy 
and energy density ; 
read per chunk, concatenating along 1 dimension (read all x, per y)
store in a netCDF file
timing: 110s/mode/y-chunk for 1440 time indicies (chunk size 63**2) 
        + 150s to load files at the beginning -> run separate jobs for each y-chunk
from comp_modal_nrjflux.py ; NJALApril 2018 '''
from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset
from scipy import signal
import time, sys
from datetime import datetime
#from xrdataset import xrdataset
from xarray import open_mfdataset as xrdataset

doverb = True

simul = 'luckyto'
ny, nx = 8, 8   # number of chunks for modal maps
#dirfil = '/ccc/scratch/cont003/gen7638/lahayen/{}/data/modemap/'.format(simul.upper())
dirfil = '/data0/project/vortex/lahaye/{}_modemap/'.format(simul)
dirfil = '/net/krypton'+dirfil # if not on libra
dirout = '/net/ruchba/local/tmp/2/lahaye/{}_modemap/'.format(simul)
filenames = [[dirfil+simul+'_modemap.{0}_{1}.nc'.format(jj,ii) for ii in range(nx)] for jj in range(ny)]
aggdims = ['eta_rho','xi_rho']
filout = simul+'_modnrj.{}.nc'

imodes = 'all'              # 'all' or numpy array

freq = 1./24. # 
forder = 4      # filter order
bpfiltype = 'butter'
methpad="gust"       # pad, gust (for scipy>=0.16.0)

grav = 9.81     # m/s^2
rho0 = 1.025    # reference density, Mg/m^3

#####   --- netCDF files    --- #####
ncr = Dataset(filenames[0][0],'r')
if imodes == 'all':
    imodes = np.arange(ncr.dimensions['mode'].size)

def createncfile(ncin,filename,imodes):
    # ncin a xarray dataset
    ncw = Dataset(filename,'w')
    for atnam,atval in ncr.attrs.items():
        if atnam != "ipx":
            ncw.setncattr(atnam, atval)
    ncw.setncattr('generating_script', sys.argv[0])
    ncw.setncattr('generated_on', datetime.today().isoformat())
	### copy dimensions and fix variables
    for dim in ['time','xi_rho','eta_rho']:
        ncw.createDimension(dim,ncin.dims[dim])
        ncw.createVariable(dim,'i2',ncin.variables[dim].dims)[:] = ncin.variables[dim][:]
    dim = 'mode'
    ncw.createDimension(dim,len(imodes))
    ncw.createVariable(dim,'i2',ncin.variables[dim].dims)[:] = ncin.variables[dim][imodes]
    for var in ['lon_rho','lat_rho','topo','ocean_time']:
        ncw.createVariable(var,'f',ncin.variables[var].dims)[:] = ncin.variables[var][:]
    var = 'mode_speed'
    ncw.createVariable(var,'f',ncin.variables[var].dims)[:] = ncin.variables[var][imodes,:,:]
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

### Start loop over y-chunks, concatenating x-chunks
if doverb:
    print("starting loop over chunks")
for jy,fnames in zip(range(ny),filenames):
    ncr = xrdataset(fnames, concat_dim='xi_rho', data_vars='minimal')
    #### compute norm
    zw = ncr.variables['zwb'][:].values
    phi = ncr.variables['w_modes'][imodes,...].values
    Nsqb = ncr.variables['Nsqb'].values
    norm_w = np.trapz(Nsqb[None,...]*phi**2,zw[None,...],axis=1) + grav*phi[:,-1,...]**2 # modified norm
    zw = np.diff(zw,axis=0)
    phi = ncr.variables['p_modes'][imodes,...].values
    norm_p = np.sum(zw[None,...]*phi**2,axis=1)
    del zw, phi, Nsqb
    
    #####   --- compute fluxes  --- #####
    dt = np.diff(ncr.variables['ocean_time'][:2])
    bb, aa = signal.butter(forder,freq*2.*dt,btype='low')
    ncw = createncfile(ncr,dirfil+filout.format(jy),imodes)
    ncw.from_files = fnames
    if doverb:
        print('netCDF file created, done initializing for chunk -- start computing')
        tmes, tmeb = time.clock(), time.time()
    ncwar = ncw.variables
    for imod,nmod in zip(range(len(imodes)),imodes):
        indoy, indox = np.where(np.isfinite(norm_p[nmod,...]))
        data = ncr.variables['p_amp'][:,nmod,...]*ncr.variables['u_amp'][:,nmod,...]\
                    *norm_p[nmod,...]*rho0      # kW/m
        ncwar['Fx'][:,imod,...] = data
        ncwar['Fx_avg'][imod,...] = np.nanmean(data,axis=0)
        data.values[:,indoy,indox] = signal.filtfilt(bb, aa, data.values[:,indoy,indox] \
                                                , axis=0, method=methpad)
        ncwar['Fx_lf'][:,imod,...] = data
        data = ncr.variables['p_amp'][:,nmod,...]*ncr.variables['v_amp'][:,nmod,...]\
                    *norm_p[nmod,...]*rho0  
        ncwar['Fy'][:,imod,...] = data
        ncwar['Fy_avg'][imod,...] = np.nanmean(data,axis=0)
        data.values[:,indoy,indox] = signal.filtfilt(bb, aa, data.values[:,indoy,indox] \
                                                , axis=0, method=methpad)
        ncwar['Fy_lf'][:,imod,...] = data
        data = (ncr.variables['u_amp'][:,nmod,...]**2 + ncr.variables['v_amp'][:,nmod,...]**2)\
                *norm_p[nmod,...]/2.*rho0       # kJ/m^2
        ncwar['ek'][:,imod,...] = data
        ncwar['ek_avg'][imod,...] = np.nanmean(data,axis=0)
        data.values[:,indoy,indox] = signal.filtfilt(bb, aa, data.values[:,indoy,indox] \
                                                , axis=0, method=methpad)
        ncwar['ek_lf'][:,imod,...] = data
        data = ncr.variables['b_amp'][:,nmod,...]**2./2.*norm_w[nmod,...]*rho0
        ncwar['ep'][:,imod,...] = data
        ncwar['ep_avg'][imod,...] = np.nanmean(data,axis=0)
        data.values[:,indoy,indox] = signal.filtfilt(bb, aa, data.values[:,indoy,indox] \
                                                , axis=0, method=methpad)
        ncwar['ep_lf'][:,imod,...] = data
        if doverb:
            print('chunk {0}, mode {1} done ; timing:'.format(jy,nmod) \
                    , time.clock()-tmes, time.time()-tmeb)
            tmes, tmeb = time.clock(), time.time()
    ncw.close()
    
