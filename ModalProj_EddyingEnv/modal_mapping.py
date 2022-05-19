''' modal_mapping.py
First, compute modes at every location in the domain, given a basic stratification (e.g. initial)
This version uses free surface calculation. Store results in a netCDF file
Second, project variables on thus-defined modes. (linearized framework, e.g. for pressure)
Treat with horizontal velocities, reduced pressure, vertical velocity and buoyancy
NJAL September 2017 '''
from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset
from Modules import *
from get_pres import get_redpres
from def_chunk import get_indchk
from dynmodes_fs import dynmodes_w
import comp_zlevs
import time, os, datetime
from mpi4py import MPI
comm = MPI.COMM_WORLD

#############################################################
#####   ---     User editable parameter section ---     #####

doverb = True
#verblev = 1

simulname = 'luckym2s'   # name of simulation
sidirname = simulname.upper().rstrip('S')+'s'
nst = 4                 # horizontal subsampling
Nmodes = 20             # number of vertical modes to use
fildir = '/ccc/scratch/cont003/gen7638/lahayen/{}/data/'.format(sidirname)
filname = 'modemap'     # suffix to create netCDF file name
npy, npx = (8,8)        # number of chunks in y, x (mpi)
ideb = 0                # it start computing modal amplitude
ireb = 509              # (start at ideb + ireb -- put 0)
ifin = 720  #1440
doappend = True    # re-open existing netCDF files and write into or not (write over)
docalmod = False    # compute modes (otherwise read into netCDF => must doaapend)
docalamp = True     # compute modal amplitudes or not
subpx = range(npx)  # put None or range(npx) to do every chunks
subpy = range(6,8)  # (put something else e.g. when appending)


#####   ---     End of parameter section    ---     #####
#########################################################
if not doappend and not docalmod:
    raise ValueError('you must append and/or compute vertical modes, but cannot do none of it')
nbproc = comm.Get_size()
if nbproc==1:
    print('Single thread: will do only one chunk')
    ipy, ipx = 5, 4                     # that is mainly when debugging or testing
else:
    if npx*npy != nbproc:
        if len(subpx)*len(subpy) == nbproc:
            print('selected chunks match with nproc: OK')
        elif not doappend:
            print('number of threads does not match internal chunking. Trying to adapt')
            npx, npy = 1, nbproc
        else:
            raise ValueError('could not select chunks -- does not match with nproc. Aborting')
    if subpx is None:
        ipx = comm.Get_rank() % npx
    else:
        ipx = subpx[comm.Get_rank() % len(subpx)]
    if subpy is None:
        ipy = comm.Get_rank() // npx
    else:
        ipy = subpy[comm.Get_rank() // len(subpx)]
        
grav = 9.81
rho0 = 1025.0

#####   load simulation, deal with chunking
simul = load(simulname,time=0)
Nx, Ny = simul.topo.shape
Nz = simul.Cs_r.__len__()
(indx,indy), (nxs,nys) = get_indchk((Nx-2,Ny-2),(npx,npy),nst)
indx = [i+1 for i in indx] 
indy = [i+1 for i in indy]
nx, ny = nxs[ipx], nys[ipy]
slix = slice(indx[ipx],indx[ipx+1],nst)
sliy = slice(indy[ipy],indy[ipy+1],nst)
topo = simul.topo[slix,sliy]
mask = simul.mask[slix,sliy]
coords = [indy[ipy],indy[ipy+1],indx[ipx],indx[ipx+1]]
coordu = [indy[ipy],indy[ipy+1],indx[ipx]-1,indx[ipx+1]+1]
coordv = [indy[ipy]-1,indy[ipy+1]+1,indx[ipx],indx[ipx+1]]
if npx*npy == 1:
    ncfilename = fildir+simulname+'_{0}.nc'.format(filname)
else:
    if npx*npy > 4:
        fildir = fildir+filname+'/'
        if not os.path.isdir(fildir):
            os.makedirs(fildir)
    ncfilename = fildir+simulname+'_{0}.{1}_{2}.nc'.format(filname,ipy,ipx)

#####   compute basic state and vertical modes
if doverb:
    print('chunk {0}   iy={2}   ix={1}   size={3}'.format((ipy,ipx),indx[ipx:ipx+2]\
            ,indy[ipy:ipy+2],(ny,nx)))
    tmes, tmeb = time.clock(), time.time()
bbase = var('buoy',simul,coord=coords).data[::nst,::nst,:]
Nsqb = var('bvf',simul,coord=coords).data[::nst,::nst,:]
zrb, zwb = toolsF.zlevs(topo,np.zeros(topo.shape),simul.hc,simul.Cs_r,simul.Cs_w)
nmodes = Nmodes
cmap = np.full((nx,ny,Nmodes),np.nan,dtype=np.float32)
wmodeb = np.full((nx,ny,Nz+1,Nmodes),np.nan)
for ix,iy in zip(*np.where(np.isfinite(mask))):
    try:
        wmode, ce, _ = dynmodes_w(Nsqb[ix,iy,:],zwb[ix,iy,:],Nmodes,fs=True)
        if wmode.shape[-1] < nmodes:
            nmodes = wmode.shape[-1]
        cmap[ix,iy,:nmodes] = ce[:nmodes]
        wmodeb[ix,iy,:,:nmodes] = wmode[:,:nmodes]
    except:
        print(ix,iy,"ca a foir'e")
if doverb:
    print('chunk {0} ended computing {1} modes ({2} required). \n\t Timing:'.format((ipy,ipx),nmodes,Nmodes)\
        ,time.clock()-tmes, time.time()-tmeb)
    tmes, tmeb = time.clock(), time.time()
if nmodes < Nmodes:
    cmap = cmap[...,:nmodes]
    wmodeb = wmodeb[...,:nmodes]
    Nmodes = nmodes
pmodeb  = np.diff(wmodeb,axis=-2)/np.diff(zwb,axis=-1)[...,None]
pmodeb = pmodeb/np.abs(pmodeb).max(axis=-2)[...,None,:]

#####   Prepare netCDF file to store results
if os.path.exists(ncfilename) and doappend:
    ncw = Dataset(ncfilename,'a')
elif doappend and not docalmod:
    raise ValueError('did not find netCDF file to read modes: aborting')
else:
    ncw = Dataset(ncfilename,'w',format='NETCDF4_CLASSIC')
    ncw.createDimension('time',None)
    ncw.createDimension('mode',nmodes)
    ncw.createDimension('s_w',Nz+1)
    ncw.createDimension('s_rho',Nz)
    ncw.createDimension('xi_rho',nx)
    ncw.createDimension('eta_rho',ny)
    # attributes
    ncw.simul = simul.simul
    ncw.generated_on = datetime.date.today().isoformat()
    ncw.generating_script = 'modal_mapping.py'
    if npx*npy > 1:
        ncw.npx = npx
        ncw.ipx = ipx
        ncw.npy = npy
        ncw.ipy = ipy
        ncw.chks_indx = indx
        ncw.chks_indy = indy
        ncw.chks_nx = nxs
        ncw.chks_ny = nys
    ncw.Sx = np.array(nxs).sum()
    ncw.Sy = np.array(nys).sum()
    ncw.nstep = nst
    # dimension- and time-invariant- variables
    ncw.createVariable('time','i4',('time',))
    ncw.createVariable('mode','i2',('mode',))[:] = np.arange(nmodes)
    ncwar = ncw.createVariable('xi_rho','i2',('xi_rho',)) 
    ncwar.longname = 'xi-indices of parent grid'
    ncwar[:] = np.arange(indx[ipx],indx[ipx+1],nst)
    ncwar = ncw.createVariable('eta_rho','i2',('eta_rho',)) 
    ncwar.longname = 'eta-indices of parent grid'
    ncwar[:] = np.arange(indy[ipy],indy[ipy+1],nst)
    ncw.createVariable('lon_rho','f',('eta_rho','xi_rho'))[:] = simul.x[slix,sliy].T
    ncw.createVariable('lat_rho','f',('eta_rho','xi_rho'))[:] = simul.y[slix,sliy].T
    ncwar = ncw.createVariable('zwb','f',('s_w','eta_rho','xi_rho'))
    ncwar.longname = "depth of sigma levels at w points for resting state"
    ncwar.units = "m"
    ncwar[:] = zwb.T
    ncwar = ncw.createVariable('zrb','f',('s_rho','eta_rho','xi_rho'))
    ncwar.longname = "depth of sigma levels at rho points for resting state"
    ncwar.units = "m"
    ncwar[:] = zrb.T
    ncwar = ncw.createVariable('topo','f',('eta_rho','xi_rho'))
    ncwar.longname = 'subsampled bathymetry'
    ncwar.units = "m"
    ncwar[:] = topo.T
    ncwar = ncw.createVariable('bbase','f',('s_rho','eta_rho','xi_rho'))
    ncwar.longname = 'basic state buoyancy'
    ncwar.unit = 'm.s^{-2}'
    ncwar[:] = bbase.T
    ncwar = ncw.createVariable('Nsqb','f',('s_w','eta_rho','xi_rho'))
    ncwar.longname = 'basic state stratification'
    ncwar.unit = 's^{-2}'
    ncwar[:] = Nsqb.T
    # vertical modes
    ncwar = ncw.createVariable('w_modes','f',('mode','s_w','eta_rho','xi_rho'))
    ncwar.longname = 'vertical velocity vertical modes, max=1-normalized'
    ncwar = ncw.createVariable('p_modes','f',('mode','s_rho','eta_rho','xi_rho'))
    ncwar.longname = 'reduced pressure p/rho0 vertical modes (times i.omega) max=1-normalized'
    ncwar = ncw.createVariable('mode_speed','f',('mode','eta_rho','xi_rho'))
    ncwar.longname = "inverse square root of eigenvalue c_n = sqrt(omega^2-f^2)/k_n"
    ncwar.units = "m/s"
    ncwar = ncw.createVariable('w_norm','f',('mode','eta_rho','xi_rho'))
    ncwar.longname = 'w modes normalization constant'
    ncwar.units = 'm^3.s^{-4}'
    ncwar = ncw.createVariable('p_norm','f',('mode','eta_rho','xi_rho'))
    ncwar.longname = 'p modes normalization constant'
    ncwar.units = 'm^5.s^{-4}'
    # other variables (time varying: mode amplitude)
    ncwar = ncw.createVariable('ocean_time','f',('time',))
    ncwar.longname = 'ocean_time or scrum_time'
    ncwar.units = 'h'
    ncwar = ncw.createVariable('w_amp','f',('time','mode','eta_rho','xi_rho'))
    ncwar.longname = 'modal amplitude, projection of w'
    ncwar = ncw.createVariable('b_amp','f',('time','mode','eta_rho','xi_rho'))
    ncwar.longname = 'modal amplitude, projection of buoyancy perturbation'
    ncwar = ncw.createVariable('p_amp','f',('time','mode','eta_rho','xi_rho'))
    ncwar.longname = 'modal amplitude, projection of reduced pressure perturbation'
    ncwar = ncw.createVariable('u_amp','f',('time','mode','eta_rho','xi_rho'))
    ncwar.longname = 'modal amplitude, projection of x-velocity'
    ncwar = ncw.createVariable('v_amp','f',('time','mode','eta_rho','xi_rho'))
    ncwar.longname = 'modal amplitude, projection of y-velocity'
ncwar = ncw.variables

### write/load modes in/from netCDF file
if docalmod:
    ncwar['w_modes'][:] = wmodeb.T
    ncwar['p_modes'][:] = pmodeb.T
    ncwar['mode_speed'][:] = cmap.T
    ncw.sync()
if doappend and not docalmod and docalamp:
    wmodeb = ncwar['w_modes'][:].T
    pmodeb = ncwar['p_modes'][:].T
    cmap = ncwar['mode_speed'][:].T
    zwb = ncwar['zwb'][:].T
    Nsqb = ncwar['Nsqb'][:].T

##############################################################################
#####  ---          Compute modal amplitudes                        ---  #####
if docalamp:
    norm_w = np.trapz(Nsqb[...,None]*wmodeb**2,zwb[...,None],axis=-2) + grav*wmodeb[...,-1,:]**2
    ncwar['w_norm'][:] = norm_w
    zwb = np.diff(zwb,axis=-1)  # zwb is now dz
    norm_p = np.nansum(zwb[...,None]*pmodeb**2,axis=-2)
    ncwar['p_norm'][:] = norm_p
    del cmap
    Nsqr = (Nsqb[...,1:]+Nsqb[...,:-1])/2.              # at rho points
    wmod = (wmodeb[:,:,1:,:] + wmodeb[:,:,:-1,:])/2.  
#####   Start loop, project modes and store
    if doverb:
        print('chunk  -- it  --  timing (cpu, ellapsed)')
    for it in range(ideb+ireb,ifin):
        simul.update(it,verbose=False)
        zeta = var('zeta',simul,coord=coords).data[::nst,::nst]
        zr = comp_zlevs.zlev_rho(topo,zeta,simul.hc,simul.Cs_r)
        #zr, zw = toolsF.zlevs(zeta,topo,simul.hc,simul.Cs_r,simul.Cs_w)
        data = tools.vinterp(var('buoy',simul,coord=coords).data[::nst,::nst,:],zrb,zr)\
                - bbase
        #res = np.trapz((data/Nsqb)[...,None]*rmodeb,zwb[...,None],axis=-2)
        res = ( np.nansum((data*zwb)[...,None]*wmod,axis=-2) \
                + grav*(data[...,-1,None]/Nsqb[...,-1,None])*wmodeb[...,-1,:] ) / norm_w
        ncwar['b_amp'][it,...] = res.T
        data = get_redpres(data,zrb,0) + grav*zeta[...,None]
        res = np.nansum((data*zwb)[...,None]*pmodeb,axis=-2) / norm_p
        ncwar['p_amp'][it,...] = res.T
        data = var('w',simul,coord=coords).data[::nst,::nst,:]
        res = ( np.nansum((data*Nsqr*zwb)[...,None]*wmod,axis=-2) \
                + grav*data[...,-1,None]*wmodeb[...,-1,:] ) / norm_w
        ncwar['w_amp'][it,...] = res.T
        data = tools.u2rho(var('u',simul,coord=coordu).data)[1:-1:nst,::nst,:]
        res = np.nansum((data*zwb)[...,None]*pmodeb,axis=-2) / norm_p
        ncwar['u_amp'][it,...] = res.T
        data = tools.v2rho(var('v',simul,coord=coordv).data)[::nst,1:-1:nst,:]
        res = np.nansum((data*zwb)[...,None]*pmodeb,axis=-2) / norm_p
        ncwar['v_amp'][it,...] = res.T
        ncwar['time'][it] = simul.time
        ncwar['ocean_time'][it] = simul.oceantime/3600.
        ncw.sync()
        if doverb:
            print('{0} -- {1:03} -- '.format((ipy,ipx),it),time.clock()-tmes,time.time()-tmeb)
            tmes, tmeb = time.clock(), time.time()
    
ncw.close()

