''' comp_btrop_fluxes.py    (formerly comp_puv_fluxes_2D.py)
compute all 2D terms (barotropic fluxes and conversion terms)
WARNING the coordinates should have one point left on every side 
(should be the case because u and v are at rho-points) for computing slope
NJAL June 2017
'''
from __future__ import division, print_function
from netCDF4 import Dataset, MFDataset
import numpy as np
import time, sys
from datetime import datetime
import scipy.signal as sig
#import R_tools_fort as toolsF
#from matplotlib import pyplot as plt
from def_chunk import get_indchk
from diff_tools import *
from mpi4py import MPI
comm = MPI.COMM_WORLD

#####  ---  Beginning of user-defined parameters  ---  #####

doverb = True

simul = 'luckyt'
if simul == "luckyt":
    sea = "_win"
else:
    sea = ""
path_base = '/data0/project/vortex/lahaye/'
path_pbot = path_base+'{0}_tseries_2Dvars/{0}_tseries_pres_iz00{1}.nc'.format(simul,sea)
path_grd = path_base+'lucky_corgrd.nc'
path_2D = path_base+'{0}_tseries_2Dvars/{0}_tseries_2Dvars{1}.nc'.format(simul,sea)
nrec = 4
nz = 80
npy = 4        # threads: number of chunks in y (fast coordinate for reading) 
if npy != comm.Get_size():
    raise ValueError('number of threads does not match')
Nmax = 4e9      # max ram per thread (more or less...)

if simul in ['lucky','luckyt']:
    dt = 1.5
elif simul in ['luckyto','luckym2','luckym2s']:
    dt = 1.0
else:
    raise ValueError('need to specify time-interval for this simulation')
freq = 1./12.42 # 
fwdt = 1.2      # relative width of pass band
flow = [1./24.]
forder = 4      # filter order
bpfiltype = 'butter'
methpad = "gust"       # pad, gust (for scipy>=0.16.0)
methpal = "gust"

outfile = path_base+'DIAG/NRJ_fluxes/{0}_btrnrj{1}.nc'.format(simul,sea)

##### ---   End of user-defined parameter section   --- #####
###############################################################################
grav = 9.81     # gravity constant (m/s^2)
rho0 = 1.025    # reference density (Mg/m^3)

ncz = Dataset(path_2D,'r')
ncgrd = Dataset(path_grd,'r')
nbeg, nend = ncz.nbeg, ncz.nend+1
nt = nend - nbeg
ny, nx = ncz.variables['lon_rho'].shape 
try:
    coord = [ncz.jmin,ncz.jmax,ncz.imin,ncz.imax]
except:
    coord = list(ncz.indy+1)+list(ncz.indx+1)       # WARNING +1 is ad-hoc

st = ncz.subsamp_step

myrank = comm.Get_rank()
(indrank,), (njs,) = get_indchk(ny,npy)
imin,imax = 0, nx
jmin, jmax = indrank[myrank:myrank+2]
nj = njs[myrank]
nchunks = int(9*nj*nx*nt/(Nmax/8./1.6)) # chunking per thread (1.6 safety factor)
(inds,), (schk,) = get_indchk(nj,nchunks)
inds = list(np.array(inds)+indrank[myrank])

# prepare file and store (add attributes and co.)
if npy > 1:
    pathwrite = outfile.replace('.nc','.{}.nc'.format(myrank))
else:
    pathwrite = outfile
if doverb:
    print('creating file',pathwrite,'for storing results')
ncw = Dataset(pathwrite,'w',format='NETCDF4_CLASSIC')
ncw.bp_freq_filt = freq
ncw.bp_freq_width = fwdt
ncw.bp_filt_type = bpfiltype
ncw.bp_filt_order = forder
ncw.bp_method = methpad
ncw.lp_freq_filt = flow
ncw.lp_filt_type = bpfiltype
ncw.lp_filt_order = forder
ncw.lp_method = methpal
ncw.nbproc = npy
ncw.rank = myrank
ncw.rank_ix = (imin,imax)
ncw.rank_iy = (jmin,jmax)
ncw.generating_script = sys.argv[0]
ncw.generated_on = datetime.today().isoformat()
ncw.createDimension('time',nt)
ncw.createDimension('xi_rho',nx)
ncw.createDimension('eta_rho',nj)
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
ncwar = ncw.createVariable('pu_bt','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"hPU: btrop. energy x-flux (backfor bandpass filtered)")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pv_bt','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"hPV: btrop. energy y-flux (backfor bandpass filtered)")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('Ct','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"Ct: topographic conversion term (backfor bandpass filtered)")
ncwar.setncattr('units','W/m^2')
ncwar = ncw.createVariable('ek_bt','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"btrop EK: barotropic kin. energy surf. density \
                    (backfor bandpass filtered)")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('ep_bt','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"btrop EP: barotropic potential energy surf. density \
                        (backfor bandpass filtered)")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('pubt_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"hPU: btrop. energy x-flux (low-pass filtered)")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pvbt_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"hPV: btrop. energy y-flux (low-pass filtered)")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('Ct_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"Ct: topographic conversion term (lowpass filtered)")
ncwar.setncattr('units','W/m^2')
ncwar = ncw.createVariable('ekbt_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"btrop EK: barotropic kin. energy surf. density \
                    (lowpass filtered)")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('epbt_lf','f',('eta_rho','xi_rho','time'))
ncwar.setncattr('long_name',"btrop EP: barotropic potential energy surf. density \
                        (lowpass filtered)")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('pubt_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"hPU: btrop. energy x-flux (time-averaged)")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('pvbt_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"hPV: btrop. energy y-flux (time-averaged)")
ncwar.setncattr('units','kW/m')
ncwar = ncw.createVariable('Ct_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"Ct: topographic conversion term (time-averaged)")
ncwar.setncattr('units','W/m^2')
ncwar = ncw.createVariable('ekbt_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"btrop EK: barotropic kin. energy surf. density \
                    (time-averaged)")
ncwar.setncattr('units','kJ/m^2')
ncwar = ncw.createVariable('epbt_avg','f',('eta_rho','xi_rho'))
ncwar.setncattr('long_name',"btrop EP: barotropic potential energy surf. density \
                        (time-averaged)")
ncwar.setncattr('units','kJ/m^2')
if doverb:
    print('file',pathwrite,'created')

# prepare filter polynomials
fpolylow = []
if bpfiltype == 'butter':
    bb, aa = sig.butter(forder,freq*2.*dt*np.array([1./fwdt,fwdt]),btype='band')
    for ff in flow:
        fpolylow.append(sig.butter(forder,ff*2*dt,btype='low'))
elif bpfiltype == 'cheby2':
    bb, aa = sig.cheby2(forder,40,freq*2.*dt*np.array([1./fwdt,fwdt]),btype='band')
    for ff in flow:
        fpolylow.append(sig.cheby2(forder,40,ff*2*dt,btype='low'))
else:
    raise ValueError('Unrecognized filter type for band-pass '+bpfiltype)
_, p, _ = sig.tf2zpk(bb,aa);         # see scipy.signal.filtfilt doc
irlen_bp = int(np.ceil(np.log(1e-9) / np.log(np.max(np.abs(p))))) 
irlen_lp = []
for bl,al in fpolylow:
    _, p, _ = sig.tf2zpk(bl,al);     # see scipy.signal.filtfilt doc
    irlen_lp.append(int(np.ceil(np.log(1e-9) / np.log(np.max(np.abs(p)))))) 

#####  ---  start computing  ---  #####
if doverb:
    print('thread {0}/{1} starts computing its {2} chunks'.format(myrank+1,npy,nchunks))
    tmes, tmeb = time.clock(), time.time()
    #tmis, tmib = time.clock(), time.time()
    ncwar = ncw.variables
for nj,jmin,jmax in zip(schk,inds[:-1],inds[1:]):
    ubar = np.full((nj,nx,nt),np.nan); vbar = np.copy(ubar); pbar = np.copy(ubar);
    prov = np.copy(ubar)
    ycoord = [int(item) for item in ncz.variables['eta_rho'][jmin:jmax:nj-1]]
    ycoord[1] += 1
    topo = ncgrd.variables['h'][ycoord[0]:ycoord[1],coord[2]:coord[3]][::st,::st]
    dhx = diff_2d_ongrid(ncgrd.variables['h'][ycoord[0]:ycoord[1],\
                        coord[2]-1:coord[3]+1],ncgrd.variables['pm'][ycoord[0]:\
                        ycoord[1],coord[2]-1:coord[3]+1],axis=1)[::st,::st]
    dhy = diff_2d_ongrid(ncgrd.variables['h'][ycoord[0]-1:ycoord[1]+1,\
                        coord[2]:coord[3]],ncgrd.variables['pn'][ycoord[0]-1:\
                        ycoord[1]+1,coord[2]:coord[3]],axis=0)[::st,::st]
    lemask = ncgrd.variables['mask_rho'][ycoord[0]:ycoord[1],coord[2]:coord[3]]\
                                                [::st,::st]
    indoy, indox = np.where(lemask)
    pbar[indoy,indox,:] = sig.filtfilt(bb,aa,sig.detrend(
                    ncz.variables['pbar'][jmin:jmax,imin:imax,:][indoy,indox,:]/1e3 + \
                    grav*rho0*ncz.variables['zeta'][jmin:jmax,imin:imax,:][indoy,indox,:], axis=-1),\
                    axis=-1,method=methpad)#,irlen=irlen_bp)
    ubar[indoy,indox,:] = sig.filtfilt(bb,aa,sig.detrend(\
                    ncz.variables['ubar'][jmin:jmax,imin:imax,:][indoy,indox,:], axis=-1),\
                    axis=-1,method=methpad)#,irlen=irlen_bp)
    vbar[indoy,indox,:] = sig.filtfilt(bb,aa,sig.detrend(\
                    ncz.variables['vbar'][jmin:jmax,imin:imax,:][indoy,indox,:], axis=-1),\
                    axis=-1,method=methpad)#,irlen=irlen_bp)
    jwin, jwax = jmin-indrank[myrank], jmax-indrank[myrank]
    ncwar['pu_bt'][jwin:jwax,imin:imax,:] = pbar*ubar*topo[...,None]
    ncwar['pv_bt'][jwin:jwax,imin:imax,:] = pbar*vbar*topo[...,None]
    ncwar['pubt_avg'][jwin:jwax,imin:imax] = np.nanmean(pbar*ubar,axis=-1)*topo
    ncwar['pvbt_avg'][jwin:jwax,imin:imax] = np.nanmean(pbar*vbar,axis=-1)*topo
    bl,al = fpolylow[0]
    prov[indoy,indox,:] = sig.filtfilt(bl,al,(pbar*ubar*topo[...,None])[indoy,indox,:],\
                    axis=-1,method=methpal)#,irlen=irlen_lp[0])
    ncwar['pubt_lf'][jwin:jwax,imin:imax,:] = prov
    prov[indoy,indox,:] = sig.filtfilt(bl,al,(pbar*vbar*topo[...,None])[indoy,indox,:],\
                    axis=-1,method=methpal)#,irlen=irlen_lp[0])
    ncwar['pvbt_lf'][jwin:jwax,imin:imax,:] = prov
    del pbar
    prov[indoy,indox,:] = Dataset(path_pbot,'r').variables['pres'] \
                                    [0,jmin:jmax,imin:imax,:][indoy,indox,:]
    prov[indoy,indox,:] = sig.filtfilt(bb, aa, sig.detrend(prov[indoy,indox,:],axis=-1) \
                    , axis=-1, method=methpad)#, irlen=irlen_bp) # p(-h)
    ncwar['Ct'][jwin:jwax,imin:imax,:] = -prov*(ubar*dhx[...,None] + vbar*dhy[...,None])
    ncwar['Ct_avg'][jwin:jwax,imin:imax] = -np.nanmean(prov*ubar,axis=-1)*dhx \
                                            - np.nanmean(prov*vbar,axis=-1)*dhy
    prov[indoy,indox,:] = sig.filtfilt(bl,al,-(prov*(ubar*dhx[...,None] \
                            + vbar*dhy[...,None]))[indoy,indox,:],\
                            axis=-1,method=methpal)#,irlen=irlen_lp[0])
    ncwar['Ct_lf'][jwin:jwax,imin:imax,:] = prov
    ncwar['ek_bt'][jwin:jwax,imin:imax,:] = rho0*topo[...,None]*(ubar**2+vbar**2)/2.
    ncwar['ekbt_avg'][jwin:jwax,imin:imax] = rho0*topo*np.nanmean(ubar**2+vbar**2,axis=-1)/2.
    prov[indoy,indox,:]= sig.filtfilt(bl,al,(ubar**2+vbar**2)[indoy,indox,:]/2.\
                        ,axis=-1,method=methpal)#,irlen=irlen_lp[0])
    ncwar['ekbt_lf'][jwin:jwax,imin:imax,:]  = rho0*topo[...,None]*prov
    del ubar, vbar
    prov[indoy,indox,:] = sig.filtfilt(bb,aa,sig.detrend(\
                    ncz.variables['zeta'][jmin:jmax,imin:imax,:][indoy,indox,:],axis=-1),\
                    axis=-1,method=methpad)#,irlen=irlen_bp)
    ncwar['ep_bt'][jwin:jwax,imin:imax,:] = grav*prov**2/2.
    ncwar['epbt_avg'][jwin:jwax,imin:imax] = grav*np.nanmean(prov**2,axis=-1)/2.
    prov[indoy,indox,:] = sig.filtfilt(bl,al,(grav*prov**2/2.)[indoy,indox,:],\
                    axis=-1,method=methpal)#,irlen=irlen_bp)
    ncwar['epbt_lf'][jwin:jwax,imin:imax,:] = prov
    if doverb:
        print('chunk {0}--{1}, time:'.format(jmin,jmax),time.clock()-tmes,time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()
ncgrd.close()
ncz.close()
ncw.close()

