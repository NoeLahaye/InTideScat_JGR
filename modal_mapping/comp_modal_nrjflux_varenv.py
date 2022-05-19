''' comp_modal_nrjflux_varenv.py
compute horizontal flux of vertically integrated energy 
and energy density ; 
store in a netCDF file
N.B.: pressure in modes is reduced pressure p/rho0 (need to multiply)
From comp_modal_nrjflux.py, NJAL April 2018 '''
from __future__ import print_function, division
import time
import numpy as np
from netCDF4 import Dataset
from scipy import signal
from xrdataset import xrdataset

doverb = True

simulpath = "LUCKYT"
season = "summer"
simul = simulpath.lower()
dirmap = '/ccc/store/cont003/gen7638/lahayen/{}/data/'.format(simulpath)
dirmod = "/ccc/scratch/cont003/gen7638/lahayen/{}/data/".format(simulpath)
dirout = dirmap
filename = simul+'_modemap_{}.nc'.format(season)
filemode = "modemap_{0}/{1}_v-modes.{2}_{3}.nc".format(season,simul,"{0}","{1}")
filout = simul+'_modenrj_{}.nc'.format(season)

imodes = 'all'              # 'all' or numpy array

freq = 1./24. # 
forder = 4      # filter order
bpfiltype = 'butter'
methpad = "gust"       # pad, gust (for scipy>=0.16.0)

grav = 9.81     # m/s^2
rho0 = 1025.    # reference density, kg/m^3
nc = Dataset(dirmod+filemode.format(0,0),"r")
npx, npy = nc.npx, nc.npy
nc.close()
filmodes = [[ dirmod+filemode.format(jy,ix) for ix in range(npx)] for jy in range(npy)]

#####   --- netCDF files    --- #####
ncr = Dataset(dirmap+filename,'r')
tmes, tmeb = time.clock(), time.time()
ncm = xrdataset(filmodes,['eta_rho','xi_rho'])
if doverb:
    print("loaded vertical modes from chunked files:",time.clock()-tmes,time.time()-tmeb)
    tmes, tmeb = time.clock(), time.time()
    
ncw = Dataset(dirout+filout,'w')
### copy dimensions and fix variables
if imodes == 'all':
    Nmodes = ncr.dimensions['mode'].size
    imodes = np.arange(Nmodes)
for dim in ['time','xi_rho','eta_rho']:
    ncw.createDimension(dim,ncr.dimensions[dim].size)
    ncw.createVariable(dim,'i2',ncr.variables[dim].dimensions)[:] = ncr.variables[dim][:]
dim = 'mode'
ncw.createDimension(dim,Nmodes)
ncw.createVariable(dim,'i2',ncr.variables[dim].dimensions)[:] = ncr.variables[dim][imodes]
for var in ['lon_rho','lat_rho','topo','ocean_time']:
    ncw.createVariable(var,'f',ncr.variables[var].dimensions)[:] = ncr.variables[var][:]
### create new variables
ncw.createVariable('Fx','f',ncr.variables['p_amp'].dimensions)
ncw.longname = 'Horizontal modal flux, x-direction'
ncw.units = 'kW/m'
ncw.createVariable('Fy','f',ncr.variables['p_amp'].dimensions)
ncw.longname = 'Horizontal modal flux, y-direction'
ncw.units = 'kW/m'
ncw.createVariable('Fx_lf','f',ncr.variables['p_amp'].dimensions)
ncw.longname = 'lowpass filtered horizontal modal flux, x-direction'
ncw.units = 'kW/m'
ncw.createVariable('Fy_lf','f',ncr.variables['p_amp'].dimensions)
ncw.longname = 'lowpass filtered horizontal modal flux, y-direction'
ncw.units = 'kW/m'
ncw.createVariable('Fx_avg','f',ncr.variables['p_amp_avg'].dimensions)
ncw.longname = 'time-averaged horizontal modal flux, x-direction'
ncw.units = 'kW/m'
ncw.createVariable('Fy_avg','f',ncr.variables['p_amp_avg'].dimensions)
ncw.longname = 'time-averaged horizontal modal flux, y-direction'
ncw.units = 'kW/m'
ncw.createVariable('ek','f',ncr.variables['p_amp'].dimensions)
ncw.longname = 'modal kinetic energy surface density'
ncw.units = 'kJ/m^2'
ncw.createVariable('ep','f',ncr.variables['p_amp'].dimensions)
ncw.longname = 'modal potential energy surface density'
ncw.units = 'kJ/m^2'
ncw.createVariable('ek_lf','f',ncr.variables['p_amp'].dimensions)
ncw.longname = 'low-pass filtered modal kinetic energy surface density'
ncw.units = 'kJ/m^2'
ncw.createVariable('ep_lf','f',ncr.variables['p_amp'].dimensions)
ncw.longname = 'low-pass filtered modal potential energy surface density'
ncw.units = 'kJ/m^2'
ncw.createVariable('ek_avg','f',ncr.variables['p_amp_avg'].dimensions)
ncw.longname = 'time-averaged modal kinetic energy surface density'
ncw.units = 'kJ/m^2'
ncw.createVariable('ep_avg','f',ncr.variables['p_amp_avg'].dimensions)
ncw.longname = 'time-averaged modal potential energy surface density'
ncw.units = 'kJ/m^2'

ncwar = ncw.variables

#### compute norm
if doverb:
    print('done with initialization, start computing')
    tmes, tmeb = time.clock(), time.time()

timelow = ncm.variables['time'].values
times = ncr.variables['time'][:]
#####   --- compute fluxes  --- #####
dt = np.diff(ncr.variables['ocean_time'][:2])
bb, aa = signal.butter(forder,freq*2.*dt,btype='low')
its, = np.where( (times >= timelow[0] ) & (times <= timelow[1]) )
for it in range(len(timelow)-1):
    if it > 0:
        its, = np.where( (times > timelow[it] ) & (times <= timelow[it+1]) )
    w_norm = ncm.variables['w_norm'].values[it:it+2,imodes,...]
    p_norm = ncm.variables['p_norm'].values[it:it+2,imodes,...]
    wt = np.array([timelow[it+1]-times[its],times[its]-timelow[it]])/(timelow[it+1]-timelow[it])
    for imod in imodes:
        norm_w = (w_norm[:,imod:imod+1,...]*wt[...,None,None]).sum(axis=0)
        norm_p = (p_norm[:,imod:imod+1,...]*wt[...,None,None]).sum(axis=0)
        data = ncr.variables['p_amp'][its,imod,...]*ncr.variables['u_amp'][its,imod,...]\
                    *norm_p*rho0/1e3      # kW/m
        ncwar['Fx'][its,imod,...] = data
        data = ncr.variables['p_amp'][its,imod,...]*ncr.variables['v_amp'][its,imod,...]\
                    *norm_p*rho0/1e3  
        ncwar['Fy'][its,imod,...] = data
        data = (ncr.variables['u_amp'][its,imod,...]**2 + ncr.variables['v_amp'][its,imod,...]**2)\
                *norm_p/2.*rho0/1e3       # kJ/m^2
        ncwar['ek'][its,imod,...] = data
        data = ncr.variables['b_amp'][its,imod,...]**2./2.*norm_w*rho0/1e3
        ncwar['ep'][its,imod,...] = data
        ncw.sync()
        if doverb:
            print('mode {} done ; timing:'.format(imod),time.clock()-tmes,time.time()-tmeb)
            tmes, tmeb = time.clock(), time.time()
ncr.close()
ncm.close()
    
print("done with computing NRJ terms -- now filtering and averaging")
for imod in imodes:
    for vname in ['Fx','Fy','ek','ep']:
        data = ncwar[vname][:,imod,:,:]
        indoy, indox = np.where(np.isfinite(data[0,...]))
        ncwar[vname+'_avg'][imod,...] = np.nanmean(data,axis=0)
        data[:,indoy,indox] = signal.filtfilt(bb,aa,data[:,indoy,indox],axis=0,method=methpad)
        ncwar[vname+'_lf'][:,imod,...] = data
    if doverb:
        print('mode {} done w/ filtering ; timing:'.format(imod), time.clock()-tmes \
                , time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()
    ncw.sync()
