''' comp_puv_fluxes.py
compute energy fluxes -- internal tides (loop on z)
same as comp_puv_fluxeb.py but read different variables in separate files
butterworth filter, backward forward hp or bp filter (e.g. at M2), compute pu and pv
then low-pass filter and mean, and store
approx vertical integration with sigma-levels at rest
load entire time series, chunk along y only (stripes), rechunk per thread if needed
WARNING: written to deal with several frequency for low-pass filtering at the end 
only for the energy flux, not for the energy
'''
from __future__ import division, print_function
from netCDF4 import Dataset
import numpy as np
import time, sys, os
import datetime
import scipy.signal as sig
#import R_tools_fort as toolsF
import comp_zlevs as zlevs
#from matplotlib import pyplot as plt
from def_chunk import get_indchk
from mpi4py import MPI
comm = MPI.COMM_WORLD

doverb = True

simul = 'luckyto'
fband = "M2"
krypton = "/data0/project/vortex/lahaye/"
path_grd = krypton+'lucky_corgrd.nc'
path_base = krypton
path_read = path_base+simul+'_tseries/'+simul+'_tseries_varname_iz00.nc'
path_2D = path_base+simul+'_tseries/'+simul+'_tseries_2Dvars.nc'
nrec = 4
nz = 80
npy = 8        # threads: number of chunks in y (fast coordinate for reading) 
if npy != comm.Get_size():
    if comm.Get_size()==1:
        print("debugging mode detected: will do 1 chunk only")
    else:
        raise ValueError('number of threads does not match')
Nmax = 3e9      # max ram per thread (more or less...)

if simul in ['lucky','luckyt']:
    dt = 1.5
else:
    dt = 1.0
if fband == "M2":
    freq = 1./12.42
elif fband == "M4":
    freq = 2./12.42 # 
elif fband == "HF":
    freq = 1/24/1.25    # roughly K1/1.25
else:
    raise ValueError("unknown frequency band")
flow = freq / 1.33  # that is engineering
fwdt = 1.2      # relative width of pass band
forder = 4      # filter order
if fband == "HF":
    bpfiltype = "high"
else:
    bpfiltype = "band"
methpad = "gust"       # pad, gust (for scipy>=0.16.0)
methpal = "gust"

outfile = path_base+'DIAG/'+'{0}_bclnrj_{1}.nc'.format(simul,fband)

##### ---   End of user-defined parameter section   --- #####
rho0 = 1.025  # reference density, Mg/m^3

#ncr = MFDataset(path_read,aggdim='s_rho')    # need NETCDF4_CLASSIC or NETCDF3
ncz = Dataset(path_2D,'r')
ncgrd = Dataset(path_grd,'r')
#ncvarz = ncz.variables['zeta']
nbeg, nend = ncz.nbeg, ncz.nend+1
nt = nend - nbeg
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
ncw.nbeg = ncz.nbeg
ncw.nend = ncz.nend
ncw.hf_freq_filt = freq
ncw.hf_filt_type = bpfiltype
if bpfiltype == 'band':
    ncw.hf_freq_width = fwdt
ncw.hf_filt_order = forder
ncw.hf_pad_meth = methpad
ncw.lp_freq_filt = flow
ncw.lp_filt_order = forder
ncw.lp_pad_meth = methpal
ncw.nbproc = npy
ncw.rank = myrank
ncw.indx_rank = (imin,imax)
ncw.indy_rank = (jmin,jmax)
ncw.ncfile_name = os.path.basename(pathwrite)
ncw.generated_on = datetime.date.today().isoformat()
ncw.generating_script = sys.argv[0]
ncw.createDimension('time',nt)
ncw.createDimension('xi_rho',nx)
ncw.createDimension('eta_rho',nj)
ncw.createDimension('s_rho',nz)
for vname in ['time','scrum_time']:
    ncwar = ncw.createVariable(vname,'f',('time'))
    for attr in ncz.variables[vname].ncattrs():
        ncwar.setncattr(attr,ncz.variables[vname].getncattr(attr))
    ncwar[:] = ncz.variables[vname][:]
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
ncwar = ncw.createVariable('bvf_avg','f',('s_rho','eta_rho','xi_rho'))
ncwar.long_name = "time-averaged buoyancy frequency at rho levels"
ncwar.units = "s^{-2}"
ncwar = ncw.createVariable('puint','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"<p'u'>: vert. int. of p'u' backfor bandpass filtered")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pvint','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"<p'v'>: vert. int. of p'v' backfor bandpass filtered")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('puint_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"time-average of <p'u'>")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pvint_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"time-average of <p'v'>")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('puint_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"low-pass filtered <p'u'> at freq 1")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pvint_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"low-pass filtered <p'v'> at freq 1")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('ekint','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"<e_k>: vert. int. of KE backfor bandpass filtered")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('epint','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"<e_p>: vert. int. of PE backfor bandpass filtered")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('ekint_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"time-averaged <e_k> at freq 1")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('epint_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"time-averaged <e_p> at freq 1")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('ekint_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"low-pass filtered <e_k> at freq 1")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('epint_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"low-pass filtered <e_p> at freq 1")
ncwar.setncattr('units','kJ/m^2')
#ncwar = ncw.createVariable('puint_lf2','f',('eta_rho','xi_rho','time'))
#ncwar.setncattr('long_name',"low-pass filtered <p'u'> at freq 2")
#ncwar.setncattr('units','W/m')
#ncwar = ncw.createVariable('pvint_lf2','f',('eta_rho','xi_rho','time'))
#ncwar.setncattr('long_name',"low-pass filtered <p'v'> at freq 2")
#ncwar.setncattr('units','W/m')
ncw.close()
if doverb:
    print('file',pathwrite,'created')

# prepare filter polynomials
fpolylow = []
if bpfiltype == 'band':
    bb, aa = sig.butter(forder, freq*2.*dt*np.array([1./fwdt,fwdt]), btype='band')
else:
    bb, aa = sig.butter(forder, freq*2.*dt, btype='high')
bl, al = sig.butter(forder,flow*2*dt,btype='low')
_, p, _ = sig.tf2zpk(bb,aa);         # see scipy.signal.filtfilt doc
irlen_bp = int(np.ceil(np.log(1e-9) / np.log(np.max(np.abs(p))))) 
_, p, _ = sig.tf2zpk(bl,al);     # see scipy.signal.filtfilt doc
irlen_lp = int(np.ceil(np.log(1e-9) / np.log(np.max(np.abs(p)))))

#####  ---  start computing  ---  #####
if doverb:
    print('thread {0}/{1} starts computing its {2} chunks'.format(myrank+1,npy,nchunks))
    tmes, tmeb = time.clock(), time.time()
    #tmis, tmib = time.clock(), time.time()
for nj,jmin,jmax in zip(schk,inds[:-1],inds[1:]):
    topo = ncgrd.variables['h'][coord[0]:coord[1],coord[2]:coord[3]]\
                                                [::st,::st][jmin:jmax,imin:imax]
    lemask = ncgrd.variables['mask_rho'][coord[0]:coord[1],coord[2]:coord[3]]\
                                                [::st,::st][jmin:jmax,imin:imax]
    indoy, indox = np.where(lemask)
    jwin, jwax = jmin-indrank[myrank], jmax-indrank[myrank]
    pbar = ncz.variables['pbar'][jmin:jmax,imin:imax,:]#[indoy,indox,:]
    
    #dzr = np.full((nx,ny,nz,nt),np.nan)
    puint = np.zeros((nj,nx,nt)); pvint = np.zeros((nj,nx,nt)); 
    ekint = np.zeros((nj,nx,nt)); epint = np.zeros((nj,nx,nt)); 
    ufilt = np.full((nj,nx,nt),np.nan); pfilt = np.copy(ufilt); 
    dzr = zlevs.dz_anyND(topo.T, np.zeros(topo.T.shape), ncz.hc, np.diff(ncz.Cs_w), nz).T
    #_, dzr = toolsF.zlevs(np.asfortranarray(topo.T),np.zeros((nx,nj)),ncz.hc,\
                #ncz.Cs_r,ncz.Cs_w)      # that's temporary z_w
    #dzr = np.diff(dzr,axis=-1).T        # dzr in m
    if doverb:
        tmis, tmib = time.clock(), time.time()
    for iz in range(nz):
        path_var = path_read.replace('00','{0:02}'.format(iz))
        var = Dataset(path_var.replace('varname','u'),'r').variables['u']
        ufilt[indoy,indox,:] = \
                sig.filtfilt(bb, aa, sig.detrend(var[0,jmin:jmax,imin:imax,:][indoy,indox,:],axis=-1) \
                            , axis=-1,method=methpad)  # ufilt is u'
                            #, axis=-1,method=methpad,irlen=irlen_bp)  # ufilt is u'
        var = Dataset(path_var.replace('varname','pres'),'r').variables['pres']
        prov = (var[0,jmin:jmax,imin:imax,:] + pbar)[indoy,indox,:]/1e3     # kPa
        #if doverb:
            #print('iz = ',iz,'rank=',myrank)
            #print(pfilt.shape,pfilt[indoy,indox,:].shape,prov.shape\
                #,prov.mean(axis=-1)[...,None].shape)
        pfilt[indoy,indox,:] = \
                sig.filtfilt(bb, aa, sig.detrend(prov, axis=-1),\
                            axis=-1, method=methpad)  # pfilt is p'
                            #axis=-1,method=methpad,irlen=irlen_bp)  # pfilt is p'
        puint += dzr[iz,...,None]*ufilt*pfilt   # x-energy flux, kW/m
        ekint += rho0*dzr[iz,...,None]*ufilt**2/2.   # half of ek (density), kJ/m^2
        var = Dataset(path_var.replace('varname','v'),'r').variables['v']
        ufilt[indoy,indox,:] = \
                sig.filtfilt(bb, aa, sig.detrend(var[0,jmin:jmax,imin:imax,:][indoy,indox,:],axis=-1)\
                            , axis=-1, method=methpad)  # ufilt is v'
                            #, axis=-1, method=methpad,irlen=irlen_bp)  # ufilt is v'
        pvint += dzr[iz,...,None]*ufilt*pfilt   # y- energy flux
        ekint += rho0*dzr[iz,...,None]*ufilt**2/2.   # finish ek
        var = Dataset(path_var.replace('varname','buoy'),'r').variables['buoy']
        prov = var[0,jmin:jmax,imin:imax,:][indoy,indox,:]
        ufilt[indoy,indox,:] = \
                sig.filtfilt(bb, aa, sig.detrend(prov, axis=-1),\
                                    axis=-1,method=methpad,irlen=irlen_bp)   # ufilt is b'
        if iz==0:       # put bvf "avg" in nsqm if mean, pfilt if lowpass
            var = Dataset(path_var.replace('varname','bvf'),'r').variables['bvf']
            nsqm = np.nanmean(var[0,jmin:jmax,imin:imax,:],axis=-1)
            #pfilt[indoy,indox,:] = sig.filtfilt(fpolylow[0][0],fpolylow[0][1],\
                                    #ncr['bvf'][0,jmin:jmax,imin:imax,:][indoy,indox,:],\
                                    #axis=-1,method=methpad,irlen=irlen_lp[0])    
        else:
            ncbvf = Dataset(path_read.replace('00','{0:02}'.format(iz-1)).replace('varname','bvf'),'r').variables['bvf']
            if iz == nz-1:
                nsqm = np.nanmean(ncbvf[0,jmin:jmax,imin:imax,:],axis=-1)
                #pfilt[indoy,indox,:] = sig.filtfilt(fpolylow[0][0],fpolylow[0][1],\
                                    #ncbvf[0,jmin:jmax,imin:imax,:][indoy,indox,:],\
                                    #axis=-1,method=methpad,irlen=irlen_lp[0])   
            else:
                var = Dataset(path_var.replace('varname','bvf'),'r').variables['bvf']
                nsqm = np.nanmean(var[0,jmin:jmax,imin:imax,:] + ncbvf\
                                    [0,jmin:jmax,imin:imax,:],axis=-1)/2.
                #pfilt[indoy,indox,:] = sig.filtfilt(fpolylow[0][0],fpolylow[0][1],\
                                    #(ncr['bvf'][0,jmin:jmax,imin:imax,:][indoy,indox,:]\
                                    #+ ncbvf[0,jmin:jmax,imin:imax,:][indoy,indox,:])/2.,\
                                    #axis=-1,method=methpad,irlen=irlen_lp[0])  
        if (nsqm[indoy,indox]==0).any():
            nsqm[indoy,indox][np.where(nsqm[indoy,indox]==0)] = 1e-10
        ncw = Dataset(pathwrite,'a')
        ncw.variables['bvf_avg'][iz,jwin:jwax,imin:imax] = nsqm
        ncw.close()
        nsqm[nsqm<=0.] = np.inf     # such that associated ep is 0
        epint += dzr[iz,...,None]*ufilt**2/nsqm[...,None]/2.     # potential NRJ density
        if not np.isfinite(puint[indoy,indox,:]*pvint[indoy,indox,:]).all():    # dirty control
            raise ValueError('il y a des nans la ou il ne devrait pas y en avoir')
        if doverb and (iz+1)%10==0:
            print('rank {0}, iz {1}:'.format(myrank,iz),time.clock()-tmis,\
                    time.time()-tmib)
            tmis, tmib = time.clock(), time.time()
    if doverb:
        print('rank {0}, chunk {1}--{2}'.format(myrank,jmin,jmax),\
                "time to compute <p'u'>, <p'v'>:",time.clock()-tmes,time.time()-tmeb,
                '; {0} values'.format(indox.size))
        tmes, tmeb = time.clock(), time.time()
    del dzr, ufilt, pfilt, nsqm
    # storing, low-pass filtering
    ncw = Dataset(pathwrite,'a')
    ncwar = ncw.variables
    ncwar['puint'][jwin:jwax,imin:imax,:] = puint
    ncwar['pvint'][jwin:jwax,imin:imax,:] = pvint
    ncwar['puint_avg'][jwin:jwax,imin:imax] = np.nanmean(puint,axis=-1)
    ncwar['pvint_avg'][jwin:jwax,imin:imax] = np.nanmean(pvint,axis=-1)
    ncwar['ekint'][jwin:jwax,imin:imax,:] = ekint
    ncwar['epint'][jwin:jwax,imin:imax,:] = epint
    ncwar['ekint_avg'][jwin:jwax,imin:imax] = np.nanmean(ekint,axis=-1)
    ncwar['epint_avg'][jwin:jwax,imin:imax] = np.nanmean(epint,axis=-1)
    
    # filtering
    prov = np.full((nj,nx,nt),np.nan,dtype=np.float32)
    prov[indoy,indox,:] = sig.filtfilt(bl, al, puint[indoy,indox,:] \
                        , axis=-1, method=methpal).astype(np.float32)
                        #, axis=-1, method=methpal, irlen=irlen_lp).astype(np.float32)
    ncwar["puint_lf"][jwin:jwax,imin:imax,:] = prov
    prov[indoy,indox,:] = sig.filtfilt(bl, al, pvint[indoy,indox,:] \
                        , axis=-1, method=methpal).astype(np.float32)
                        #, axis=-1, method=methpal, irlen=irlen_lp).astype(np.float32)
    ncwar['pvint_lf'][jwin:jwax,imin:imax,:] = prov
    prov[indoy,indox,:] = sig.filtfilt(bl, al, ekint[indoy,indox,:] \
                        , axis=-1, method=methpal).astype(np.float32)
                        #, axis=-1, method=methpal, irlen=irlen_lp).astype(np.float32)
    ncwar['ekint_lf'][jwin:jwax,imin:imax,:] = prov
    prov[indoy,indox,:] = sig.filtfilt(bl, al, epint[indoy,indox,:], axis=-1 \
                                , method=methpal).astype(np.float32)
                                #, method=methpal, irlen=irlen_lp).astype(np.float32)
    ncwar['epint_lf'][jwin:jwax,imin:imax,:] = prov
    del prov
    ncw.close()

    if doverb:
        print('rank {0}, chunk {1}--{2}'.format(myrank,jmin,jmax),\
            "time to filter and store:",time.clock()-tmes,time.time()-tmeb,\
            'written in slab {0}-{1}'.format(jwin,jwax))
        tmes, tmeb = time.clock(), time.time()

ncgrd.close()
ncz.close()

