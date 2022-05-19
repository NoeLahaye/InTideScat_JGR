""" comp_pw_flux.py
compute vertical energy flux on every sigma levels
barotropic component of w is neglected, barotropic pressure g*rho0*zeta is removed
fields are high passed or band passed before the product
from comp_puv_fluxes.py, removed mpi instructions 
NJAL April 2018 """

from __future__ import division, print_function
from netCDF4 import Dataset, MFDataset
import numpy as np
import time
from scipy import signal
import R_tools_fort as toolsF
from def_chunk import get_indchk
from mpi4py import MPI
comm = MPI.COMM_WORLD

doverb = True

simul = 'luckyto'
fband = "M2"
path_grd = '/net/ruchba/local/tmp/2/lahaye/prep_LUCKYTO/lucky_grd.nc'
path_base = '/net/krypton/data0/project/vortex/lahaye/'
path_read = path_base+simul+'_tseries/'+simul+'_tseries_varname_iz00.nc'
path_2D = path_base+simul+'_tseries/'+simul+'_tseries_2Dvars.nc'
nrec = 4
zlevs = [1, 20, 40, 60]
nst = 2             # subsampling step for unfiltered fields (in time)
nsl = 4            # subsampling step for filtered fields
itmin, itmax = (720, 1079)   # min, max time indices to do the computation (time step of simus)
nz = len(zlevs)
npy = 8        # threads: number of chunks in y (slow coordinate for reading) 
if npy != comm.Get_size():
    if comm.Get_size()==1:
        print("debugging mode detected: will do 1 chunk only")
    else:
        raise ValueError('number of threads does not match')
Nmax = 4e9      # max ram per thread (more or less...)

if simul in ['lucky','luckyt']:
    dt = 1.5
else:
    dt = 1.0
if fband == "M2":
    freq = 1./12.42
    flow = 1./12.
elif fband == "M4":
    freq = 2./12.42 # 
    flow = 1./12.
else:
    raise ValueError("unknown frequency band")
fwdt = 1.2      # relative width of pass band
forder = 4      # filter order
bpfiltype = 'butter'
methpad = "gust"       # pad, gust (for scipy>=0.16.0)

outfile = path_base+'DIAG/NRJ_fluxes/'+'{0}_pw_fluxes_{1}.nc'.format(simul,fband)

##### ---   End of user-defined parameter section   --- #####
rho0 = 1.025  # reference density, Mg/m^3

#ncr = MFDataset(path_read,aggdim='s_rho')    # need NETCDF4_CLASSIC or NETCDF3
ncz = Dataset(path_2D,'r')
ncgrd = Dataset(path_grd,'r')
#ncvarz = ncz.variables['zeta']
times = ncz.variables["time"][:]
its, = np.where( (times>=itmin) & (times<=itmax) )
it1, it2 = its[0], its[-1]+1
times = times[its]
tsub = times[nst//2::nst]
tsul = times[nsl//2::nsl]
nt, nts, ntl = len(times), len(tsub), len(tsul)
ny, nx = ncz.variables['lon_rho'].shape 
coord = list(ncz.indy) + list(ncz.indx)
st = ncz.subsamp_step

myrank = comm.Get_rank()
(indrank,), (njs,) = get_indchk(ny,npy)
imin,imax = 0, nx
jmin, jmax = indrank[myrank:myrank+2]
nj = njs[myrank]
nchunks = int(8*nj*nx*nt/(Nmax/8.)) # chunking per thread 
nchunks += 1 # safer than ...
(inds,), (schk,) = get_indchk(nj,nchunks)
inds = list(np.array(inds)+indrank[myrank])
print('rank',myrank,'imin,imax:',imin,imax,'jmin,jmax',jmin,jmax)

# prepare file and store (add attributes and co.)
pathwrite = outfile.replace('.nc','.{}.nc'.format(myrank))
if doverb:
    print('creating file',pathwrite,'for storing results')
ncw = Dataset(pathwrite,'w',format='NETCDF4_CLASSIC')
ncw.bp_freq_filt = freq
ncw.bp_freq_width = fwdt
ncw.bp_filt_type = bpfiltype
ncw.bp_filt_order = forder
ncw.lp_freq_filt = flow
ncw.lp_filt_type = bpfiltype
ncw.lp_filt_order = forder
ncw.nbproc = npy
ncw.rank = myrank
ncw.rank_ix = (imin,imax)
ncw.rank_iy = (jmin,jmax)
ncw.createDimension('time',nts)
ncw.createDimension('tlow',ntl)
ncw.createDimension('xi_rho',nx)
ncw.createDimension('eta_rho',nj)
ncw.createDimension('s_rho',len(zlevs))
for vname in 'xi_rho','eta_rho':
    ncwar = ncw.createVariable(vname,'i',(vname,))
    for attr in ncz.variables[vname].ncattrs():
        ncwar.setncattr(attr,ncz.variables[vname].getncattr(attr))
ncw.variables['xi_rho'][:] = ncz.variables['xi_rho'][:]
ncw.variables['eta_rho'][:] = ncz.variables['eta_rho'][jmin:jmax]
for vname in 'lon_rho','lat_rho':
    ncwar = ncw.createVariable(vname,'f',('eta_rho','xi_rho'))
    for attr in ncz.variables[vname].ncattrs():
        ncwar.setncattr(attr,ncz.variables[vname].getncattr(attr))
    ncwar[:] = ncz.variables[vname][jmin:jmax,:]
ncwar = ncw.createVariable('time','i2',('time',))
ncwar[:] = times[nst//2::nst]
ncwar.long_name = "time indices for unfiltered fields"
ncwar = ncw.createVariable('tlow','i2',('tlow',))
ncwar[:] = times[nsl//2::nsl]
ncwar.long_name = "time indices for filtered fields"
ncwar = ncw.createVariable('bvf_avg','f',('s_rho','eta_rho','xi_rho'))
ncwar.long_name = "time-averaged buoyancy frequency at rho levels"
ncwar.units = "s^{-2}"
ncwar = ncw.createVariable('pw','f',('s_rho','eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"p'w': vertical NRJ flux")
ncwar.setncattr('units','W/m^2')
ncwar = ncw.createVariable('pw_avg','f',('s_rho','eta_rho','xi_rho'))
ncwar.setncattr('long_name',"time-average of p'w'")
ncwar.setncattr('units','W/m^2')
ncwar = ncw.createVariable('pw_lf','f',('s_rho','eta_rho','xi_rho','tlow'))
ncwar.setncattr('long_name',"low-pass filtered p'w' lp_freq_filt")
ncwar.setncattr('units','W/m^2')
ncw.close()
if doverb:
    print('file',pathwrite,'created')

# prepare filter polynomials
bb, aa = signal.butter(forder, freq*2.*dt*np.array([1./fwdt,fwdt]), btype='band')
bl, al = signal.butter(forder, flow*2*dt, btype='low')
_, p, _ = signal.tf2zpk(bb,aa);         # see scipy.signal.filtfilt doc
irlen_bp = int(np.ceil(np.log(1e-9) / np.log(np.max(np.abs(p)))) ) 
_, p, _ = signal.tf2zpk(bl,al);     
irlen_lp = int(np.ceil(np.log(1e-9) / np.log(np.max(np.abs(p)))) ) 

#####  ---  start computing  ---  #####
if doverb:
    print('thread {0}/{1} starts computing its {2} chunks'.format(myrank+1,npy,nchunks))
    tmes, tmeb = time.clock(), time.time()
    #tmis, tmib = time.clock(), time.time()
for isz in range(len(zlevs)):
    iz = zlevs[isz]
    path_var = path_read.replace('00','{0:02}'.format(iz))
    var = Dataset(path_var.replace('varname','w'),'r').variables['w']
    for nj,jmin,jmax in zip(schk,inds[:-1],inds[1:]):
        topo = ncgrd.variables['h'][coord[0]:coord[1],coord[2]:coord[3]]\
                                                [::st,::st][jmin:jmax,imin:imax]
        lemask = ncgrd.variables['mask_rho'][coord[0]:coord[1],coord[2]:coord[3]]\
                                                [::st,::st][jmin:jmax,imin:imax]
        indoy, indox = np.where(lemask)
        #pbar = ncz.variables['pbar'][jmin:jmax,imin:imax,:]#[indoy,indox,:]
    
        #dzr = np.full((nx,ny,nz,nt),np.nan)
        pw = np.zeros((nj,nx,nt)); 
        _, dzr = toolsF.zlevs(np.asfortranarray(topo.T),np.zeros((nx,nj)),ncz.hc,\
                ncz.Cs_r,ncz.Cs_w)      # that's temporary z_w
        dzr = np.diff(dzr,axis=-1).T        # dzr in m
        if doverb:
            tmis, tmib = time.clock(), time.time()
        pw[indoy,indox,:] = signal.filtfilt(bb,aa,\
                                var[0,jmin:jmax,imin:imax,it1:it2][indoy,indox,:],\
                                axis=-1,method=methpad,irlen=irlen_bp)  
        var = Dataset(path_var.replace('varname','pres'),'r').variables['pres']
        #prov = (var[0,jmin:jmax,imin:imax,it1:it2] + pbar[...,it1:it2])[indoy,indox,:]
        prov = var[0,jmin:jmax,imin:imax,it1:it2][indoy,indox,:]
        pw[indoy,indox,:] *= signal.filtfilt(bb,aa,prov-prov.mean(axis=-1)[...,None],\
                                    axis=-1,method=methpad,irlen=irlen_bp) 
        if not np.isfinite(pw[indoy,indox,:]).all():    # dirty control
            raise ValueError('il y a des nans la ou il ne devrait pas y en avoir')
        del dzr
        # storing, low-pass filtering
        jwin, jwax = jmin-indrank[myrank], jmax-indrank[myrank]
        ncw = Dataset(pathwrite,'a')
        ncwar = ncw.variables
        ncwar['pw'][isz,jwin:jwax,imin:imax,:] = pw[:,:,nst//2::nst]
        ncwar['pw_avg'][isz,jwin:jwax,imin:imax] = np.nanmean(pw,axis=-1)
    
        # filtering
        prov = np.full((nj,nx,nt),np.nan,dtype=np.float32)
        prov[indoy,indox,:] = signal.filtfilt(bl, al, pw[indoy,indox,:] \
                        , axis=-1, method=methpad, irlen=irlen_lp).astype(np.float32)
        ncwar["pw_lf"][isz,jwin:jwax,imin:imax,:] = prov[:,:,nsl//2::nsl]
        del prov
        ncw.close()

        if doverb:
            print('rank {0}, iz {1}, chunk {2}--{3}'.format(myrank,iz,jmin,jmax),\
                "time to filter and store:",time.clock()-tmes,time.time()-tmeb,\
                'written in slab {0}-{1}'.format(jwin,jwax))
            tmes, tmeb = time.clock(), time.time()

ncgrd.close()
ncz.close()


