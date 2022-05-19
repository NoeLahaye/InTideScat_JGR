''' calc_modal_surf_spectra.py
compute frequency PSD of modal surface (or any sigma level) fields
from calc_hke_spectra_hslabs.py ; NJAL April 2018  '''
from __future__ import print_function, division
import numpy as np
import scipy.signal as sig
import sys, os, time
from datetime import datetime
from netCDF4 import Dataset
from def_chunk import get_indchk

iz = 0  # this is not implemented for now: do surface by default
simpath = "LUCKYM2"
simul = simpath.lower()
path_data = "/data0/project/vortex/lahaye/{}_modemap/".format(simul)
fnamein = path_data+"{}_modemap.nc".format(simul)
fnamout = path_data+"{}_modal_surf_spectra.nc".format(simul)
#fnamout = "/net/ruchba/local/tmp/2/lahaye/DIAG/{}_modal_surf_spectra.nc".format(simul)
doappend = True

imodes = "all"
whatfield = "hke"  # "sse", "hke"

ncr = Dataset(fnamein,"r")
oceantime = ncr.variables['ocean_time'][:]
dt = np.diff(oceantime[:2])[0]    # time step in hour
Lwin = 7     # length of welch's window (in day)
Nt = oceantime.size
nwindow = int(Lwin*24./dt)

grav = 9.81
rho0 = 1.025

if imodes == "all":
    Nmodes = ncr.dimensions["mode"].size
    imodes = np.arange(Nmodes)
else:
    Nmodes = len(imodes)

### make first calculation to get frequency and test, before initializing netCDF file
vname = "sp"+whatfield
if whatfield == "sse":
    longname = "Sea-surface elevation modal PSD"
    units = "m^2.day"
elif whatfield == "hke":
    longname = "Sea-surface HKE modal PSD"
    units = "kJ.day.m^{-2}"
else:
    raise  ValueError('Unrecognised name for field to compute')
freq, _ = sig.welch(oceantime, fs=24./dt, nperseg=nwindow)
Nf = len(freq)

######   Prepare netCDF for storing results #####
if os.path.exists(fnamout) and doappend:
    ncw = Dataset(fnamout,'r+')
else:
    ncw = Dataset(fnamout,'w')
    for attr in ncr.ncattrs():
        ncw.setncattr(attr,ncr.getncattr(attr))
    ncw.generated_on = datetime.today().isoformat()
    ncw.generating_script = sys.argv[0]
    ncw.from_file = fnamein
    ncw.welch_window_length = str(Lwin)+" days"
# createdimensions
if 'freq' not in ncw.dimensions:
    ncw.createDimension('freq',Nf)
if 'freq' not in ncw.variables:
    lavar = ncw.createVariable('freq','f',('freq',))
    lavar.long_name = 'frequency'
    lavar.units = 'day^{-1}'
    lavar[:] = freq[:]
for vnam in ["lon_rho","lat_rho"]:
    if vnam not in ncw.variables:
        for dim in ncr.variables[vnam].dimensions:
            if dim not in ncw.dimensions:
                ncw.createDimension(dim,ncr.dimensions[dim].size)
            if dim not in ncw.variables:
                ncwar = ncw.createVariable(dim,ncr.variables[dim].dtype,(dim,))
                ncwar[:] = ncr.variables[dim][:]
                for attr in ncr.variables[dim].ncattrs():
                    ncwar.setncattr(attr,ncr.variables[dim].getncattr(attr))
        ncwar = ncw.createVariable(vnam,ncr.variables[vnam].dtype,ncr.variables[vnam].dimensions)
        ncwar[:] = ncr.variables[vnam][:]
        for attr in ncr.variables[vnam].ncattrs():
            ncwar.setncattr(attr,ncr.variables[vnam].getncattr(attr))
if "mode" not in ncw.variables:
    if "mode" not in ncw.dimensions:
        ncw.createDimension("mode",Nmodes)
    lavar = ncw.createVariable('mode','i2',('mode',))[:] = ncr.variables["mode"][:Nmodes]
if vname not in ncw.variables:
    lavar = ncw.createVariable(vname,'f',('mode','eta_rho','xi_rho','freq'))
    lavar.long_name = longname
    lavar.units = units
ncwar = ncw.variables[vname]
print("netCDF file is ready")

Ny, Nx = ncr.variables["p_modes"].shape[-2:]
(inds,), (sch,) = get_indchk(Ny,4)

tmes, tmeb = time.clock(), time.time()
tmeg, tmeh = tmes, tmeb
for imod in imodes:
#for imod in [0]:
    for ii in range(len(sch)):
        i1, i2 = inds[ii], inds[ii+1]
        ioy, iox = np.where(np.isfinite(ncr.variables["p_modes"][imod,-1,i1:i2,:]))
        if whatfield == "sse":
            data = ncr.variables["p_modes"][imod,-1:,i1:i2,:] \
                        * ncr.variables["p_amp"][:,imod,i1:i2,:]/grav
            tread, treab = time.clock() - tmes, time.time() - tmeb
            _, psd = sig.welch(data[:,ioy,iox], fs=24./dt, nperseg=nwindow \
                            , detrend='linear', axis=0)
            tcalc, tcalb = time.clock() - tmes - tread, time.time() - treab - tmeb
        elif whatfield == "hke":
            data = ncr.variables["p_modes"][imod,-1:,i1:i2,:] \
                        * ncr.variables["u_amp"][:,imod,i1:i2,:]
            tread, treab = time.clock() - tmes, time.time() - tmeb
            _, psd = sig.welch(data[:,ioy,iox], fs=24./dt, nperseg=nwindow \
                            , detrend='linear', axis=0)
            tcalc, tcalb = time.clock()-tread, time.time() - treab - tmeb
            data = ncr.variables["p_modes"][imod,-1:,i1:i2,:] \
                        * ncr.variables["v_amp"][:,imod,i1:i2,:]
            tread = time.clock() - tcalc
            treab = time.time() - tmeb - tcalb
            _, data = sig.welch(data[:,ioy,iox], fs=24./dt, nperseg=nwindow \
                            , detrend='linear', axis=0)
            psd = (psd + data)*rho0/2.
            tcalc = time.clock() - tmes - tread 
            tcalb = time.time() - tmeb - treab 
        data = np.full((sch[ii],Nx,Nf),np.nan)      # I don't know why I must use this
        data[ioy,iox,:] = psd.T                     # temporary array, otherwise ncwar
        ncwar[imod,i1:i2,:,:] = data                # remains empty...
        print("imod {0}, end with chunk #{1}: [{2}-{3}] ; timing:".format(imod,ii,i1,i2), \
                time.clock()-tmes, time.time()-tmeb)
        print("  --- reading:", tread, treab)
        print("  --- computing:", tcalc, tcalb)
        print("  --- writing:", time.clock() - tmes - tread - tcalc \
                        , time.time() - tmeb - treab - tcalb)
        tmes, tmeb = time.clock(), time.time()
    ncw.sync()
ncr.close()
ncw.close()
print("end computing, took", time.clock()-tmeg, time.time()-tmeh)

