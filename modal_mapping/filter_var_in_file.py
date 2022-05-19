""" filter_var_in_file.py
load variable in a given netCDF file containing time series on sigma level 
(variable shape 1, Ny, Nx, Nt), compute and store filtered and averaged fields
create a netCDF and copy most of the dimensions and dimension variables from the original
will store a subsampled version of the filtered field, along the dimension as the filtering procedure (relevant for low-pass filtering)
adapted from filter_var_in_file.py, NJAL February 2018 """
from __future__ import print_function, division
import numpy as np
import time, datetime
import os, sys
from netCDF4 import Dataset
import scipy.signal as sig
from def_chunk import get_indchk

doverb = True

####     User-defined parameters section
simul = 'luckyt'
varname = 'buoy'
pathbase = "/".join([os.getenv("SCRATCHDIR"),simul.upper(),"data/tseries/"])
pathread = pathbase+'{0}_tseries_{1}_iz{2}.nc'.format(simul,varname,"{:02d}")
pathstor = pathbase+"{0}_lowf_{1}.nc".format(simul,varname)
transperr = False   # data are horizontally transposed (error on time series)
                    # this is the case on LUCKYT from 2705 to 2945

if varname == 'bvf':
    Nz = 79
else:
    Nz = 80
Nmax = 2e9      # max size available in ram 
nst = 16        # temporal subsampling for storing values

fcut = 1/48.    # good practice au pif: fcut = fsubsampout/2 (1/nst/dt/2)
fmod = 'low'


####    Load netCDF to read from, create netCDF to write in, copy stuffs
ncr = Dataset(pathread.format(0),'r')
ncw = Dataset(pathstor,'w')
for dim in ncr.dimensions:
    if dim == 'time':
        Nt = ncr.dimensions[dim].size
        (its,), (nt,) = get_indchk(Nt,1,nst)      # center time-subsample
        nt = nt[0]
        ncw.createDimension(dim,nt)
        ncw.createVariable(dim,'i',(dim,))[:] = ncr.variables[dim][its[0]::nst]
    elif dim in ["s_w","s_rho"]:
        ncw.createDimension(dim,Nz)
        ncw.createVariable(dim,'i',(dim,))[:] = np.arange(Nz)
    else:
        ncw.createDimension(dim,ncr.dimensions[dim].__len__())
        ncw.createVariable(dim,'i',(dim,))[:] = ncr.variables[dim][:]
    # special case: add vertical
for var in ['scrum_time','lon_rho','lat_rho']:
    if var in ncr.variables:
        ncwar = ncw.createVariable(var,'f',ncr.variables[var].dimensions)
        if 'time' in ncwar.dimensions:
            ncwar[:] = ncr.variables[var][its[0]::nst]
        else:
            ncwar[:] = ncr.variables[var][:]
for attr in ncr.ncattrs():
    if attr not in ['generated_on','generating_script']:
        ncw.setncattr(attr,ncr.getncattr(attr))
lavar = ncr.variables[varname]
ncw.generated_on = datetime.date.today().isoformat()
ncw.generating_script = sys.argv[0]
ncw.from_ncfile = pathread.split()[-1]
dims = lavar.dimensions
ncwar = ncw.createVariable(varname+'_{}f'.format(fmod),'f',dims)
for attr in lavar.ncattrs():
    ncwar.setncattr(attr,lavar.getncattr(attr))
    ncwar.description = fmod+'-pass filtered at frequency {0:.2e} hour^-1'.format(fcut)
ncwag = ncw.createVariable(varname+'_avg','f',dims[:-1])
for attr in lavar.ncattrs():
    ncwag.setncattr(attr,lavar.getncattr(attr))
    ncwag.description = 'time averaged over the entire time series'
dt = np.diff(ncr.variables['scrum_time'][:2])[0]  
if ncr.variables['scrum_time'].units.strip() == 's':
    dt /= 3600.     # to hours
Ni,Nj = lavar.shape[1:-1]
ncr.close()
if doverb:
    print("done with creating netCDF file",pathstor)

#####   Define chunking (memory purpose)
nchunks = int(8*Ni*Nj*Nt/Nmax*4)    # *4 is ad hoc
(inds,),(schk,) = get_indchk(Ni,nchunks)
if doverb:
    print('will use {} chunks'.format(nchunks))
    tmes, tmeb = time.clock(), time.time()

bb, aa = sig.butter(4,fcut*2*dt,btype=fmod)

#####   Start loop over chunks: filter and store
for iz in range(Nz):
    ncr = Dataset(pathread.format(iz),'r')
    lavar = ncr.variables[varname][:]
    ncr.close()
    if doverb: print("done reading")
    for ich in range(nchunks):
        i1,i2 = inds[ich:ich+2]
        prov = lavar[0,i1:i2,...]
        if transperr:
            ncwag[iz,:,i1:i2] = np.nanmean(prov,axis=-1).T
        else:
            ncwag[iz,i1:i2,:] = np.nanmean(prov,axis=-1)
        print('stored mean, start filtering',prov.shape)
        prov = sig.filtfilt(bb,aa,prov,axis=-1,method='gust')[...,its[0]::nst]
        if transperr:
            ncwar[iz,:,i1:i2,:] = np.rollaxis(prov,1)
        else:
            ncwar[iz,i1:i2,:,:] = prov
        print('stored lowpass')
        if doverb:
            print('iz = {0}, iy = {1} -- timing:'.format(iz,ich),time.clock()-tmes,time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()

ncw.close()

