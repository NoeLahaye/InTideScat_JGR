''' modal_mapping.py
First, compute modes at every location in the domain, given a basic stratification (e.g. initial)
This version uses free surface calculation. Store results in a netCDF file
Second, project variables on thus-defined modes. (linearized framework, e.g. for pressure)
Treat with horizontal velocities, reduced pressure, vertical velocity and buoyancy
WARNING: because of free surface and associated projection implying BVF at the surface, the modal buoyancy looks pretty bad
NJAL September 2017 '''
from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset
from Modules import *
from get_pres import get_redpres
from def_chunk import get_indchk
from get_modes_interp_file import get_modes_interp_file
import comp_zlevs
import time, os, datetime, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD

#############################################################
#####   ---     User editable parameter section ---     #####

doverb = True
#verblev = 1

simulname = 'luckyt'   # name of simulation
sidirname = simulname.upper()
nst = 4                 # horizontal subsampling
Nmodes = 10             # number of vertical modes to use
fildir = '/ccc/scratch/cont003/gen7638/lahayen/{}/data/'.format(sidirname)
filname = 'modemap_winter'  # suffix to create netCDF file name (modal amplitudes) of dir.
filmode = "v-modes"     # for mode structure
bastate_filename = fildir+"{}_subsamp_stratif.nc".format(simulname)
npy, npx = (8,8)        # number of chunks in y, x (mpi)
ideb = 2705 #5549       # it start computing modal amplitude
ireb = 0                # (start at ideb + ireb -- put 0)
ifin = 2946 #5791 #
doappend = False    # re-open existing netCDF files and write into or not (write over)
dosavmod = True     # write modes in netCDF file
subpx = range(npx)  # put None or range(npx) to do every chunks
subpy = range(npy)  # (put something else e.g. when appending)

# Take LUCKYT patches if necessary
if simulname == "luckyt" and ideb >= 5472 and ifin < 5792:
    dorep = True
    repstr = ['vicc/LUCKYT/HIS','gulaj/LUCKYT']
    bastate_filename = bastate_filename.rstrip('if.nc')+'_sum.nc'
else:
    dorep = False
    if simulname == 'luckyt':
        bastate_filename = bastate_filename.rstrip('if.nc')+'_win.nc'


#####   ---     End of parameter section    ---     #####
#########################################################
nbproc = comm.Get_size()
if nbproc==1:
    print('Single thread: will do only one chunk')
    if len(sys.argv) > 1:       # use chunk indices from args
        ipy, ipx = int(sys.argv[1]), int(sys.argv[2]) 
    else:
        ipy, ipx = 5, 4         # debugging, working on a particular chunk
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
    ncfilemode = fildir+simulname+'_{0}.nc'.format(filmode)
else:
    if npx*npy > 4:
        fildir = fildir+filname+'/'
        if not os.path.isdir(fildir) and comm.Get_rank() == 0:
            os.makedirs(fildir)
    ncfilename = fildir+simulname+'_{0}.{1}_{2}.nc'.format("modemap",ipy,ipx)
    ncfilemode = fildir+simulname+'_{0}.{1}_{2}.nc'.format("v-modes",ipy,ipx)

#####   load files for basic state and prepare interpolation
if doverb:
    print('chunk {0}   iy={2}   ix={1}   size={3}'.format((ipy,ipx),indx[ipx:ipx+2]\
            ,indy[ipy:ipy+2],(ny,nx)))
    tmes, tmeb = time.clock(), time.time()
nc = Dataset(bastate_filename)
xi = nc.variables['xi_rho'][:]
i1 = np.where(xi>indx[ipx])[0][0]-1
i2 = np.where(xi<indx[ipx+1])[0][-1]+2
eta = nc.variables['eta_rho'][:]
j1 = np.where(eta>indy[ipy])[0][0]-1
j2 = np.where(eta<indy[ipy+1])[0][-1]+2
lon = nc.variables['lon_rho'][j1:j2,i1:i2].T
lat = nc.variables['lat_rho'][j1:j2,i1:i2].T
Nysub, Nxsub = nc.variables['h'].shape
times = nc.variables['time'][:]
nc.close()
tmes, tmeb = time.clock(), time.time()
elem, coef = sm.get_tri_coef(lon,lat,simul.x[slix,sliy],simul.y[slix,sliy])
eli, elj = np.unravel_index(elem,lon.shape)
elem = np.ravel_multi_index((eli+i1,elj+j1),(Nxsub,Nysub)) # elems indices in entire grid
print("triangulation",time.clock()-tmes,time.time()-tmeb)
xi, eta = xi[i1:i2], eta[j1:j2]
nbx, nby = xi.size, eta.size

zr, zw = toolsF.zlevs(simul.topo[slix,sliy], np.zeros((nx,ny)), simul.hc, simul.Cs_r \
                        , simul.Cs_w)   # z-grid at computed points in chunk

nmodes = Nmodes


#####   Prepare netCDF files to store results: mode properties (2D only)
if os.path.exists(ncfilemode) and doappend:
    ncw = Dataset(ncfilemode,'a')
else:
    ncw = Dataset(ncfilemode,'w',format='NETCDF4_CLASSIC')
    ncw.createDimension('time',None)
    ncw.createDimension('mode',nmodes)
    ncw.createDimension('xi_rho',nx)
    ncw.createDimension('eta_rho',ny)
    # attributes
    ncw.simul = simul.simul
    ncw.generated_on = datetime.date.today().isoformat()
    ncw.generating_script = sys.argv[0]
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
    ncw.createVariable('mode_speed','f',('time','mode','eta_rho','xi_rho'))
    ncwar.longname = "inverse square root of eigenvalue c_n = sqrt(omega^2-f^2)/k_n"
    ncwar.units = "m/s"
    ncwar = ncw.createVariable('w_norm','f',('time','mode','eta_rho','xi_rho'))
    ncwar.longname = 'w modes normalization constant'
    ncwar.units = 'm^3.s^{-4}'
    ncwar = ncw.createVariable('p_norm','f',('time','mode','eta_rho','xi_rho'))
    ncwar.longname = 'p modes normalization constant'
    ncwar.units = 'm^5.s^{-4}'
ncw.close()

#####   Prepare netCDF files to store results: amplitudes
if os.path.exists(ncfilename) and doappend:
    ncw = Dataset(ncfilename,'a')
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
    ncw.generating_script = sys.argv[0]
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
    ncwar[:] = zw.T
    ncwar = ncw.createVariable('zrb','f',('s_rho','eta_rho','xi_rho'))
    ncwar.longname = "depth of sigma levels at rho points for resting state"
    ncwar.units = "m"
    ncwar[:] = zr.T
    ncwar = ncw.createVariable('topo','f',('eta_rho','xi_rho'))
    ncwar.longname = 'subsampled bathymetry'
    ncwar.units = "m"
    ncwar[:] = topo.T
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
ncw.close()


##############################################################################
#####  ---          Compute modes and modal amplitudes: loop        ---  #####
cmap = np.zeros((nx,ny,Nmodes),dtype=np.float32)
wmodeb = np.zeros((2,nx,ny,Nz+1,Nmodes),dtype=np.float32)
pmodeb = np.zeros((2,nx,ny,Nz,Nmodes),dtype=np.float32)
norm_w = np.zeros((2,nx,ny,Nmodes),dtype=np.float32)
norm_p = np.zeros((2,nx,ny,Nmodes),dtype=np.float32)
Nsqb = np.zeros((2,nx,ny,Nz+1),dtype=np.float32)
bbase = np.zeros((2,nx,ny,Nz),dtype=np.float32)
# First computation of modes
it0 = np.where(times <= (ideb+ireb))[0][-1]
(wmodeb[1,...], pmodeb[1,...]), (cmap, norm_w[1,...], norm_p[1,...]) \
        , (Nsqb[1,...], bbase[1,...]) = get_modes_interp_file( \
                        bastate_filename, it0, Nmodes \
                        , elem, coef, zw, zr, mask, doverb=True )
ncw = Dataset(ncfilemode,'r+')
ncw.variables['time'][it0] = times[it0]
ncw.variables['mode_speed'][it0,...] = cmap.T
ncw.variables['w_norm'][it0,...] = norm_w[1,...].T
ncw.variables['p_norm'][it0,...] = norm_p[1,...].T
ncw.close()
tits = [-1, it0]
tims = [0, times[tits[1]]]
del cmap
wmod = (wmodeb[:,:,1:,:] + wmodeb[:,:,:-1,:])/2.  
dzw = np.diff(zw,axis=-1)

#####   Start loop, project modes and store
if doverb:
    print('chunk  -- it  --  timing (cpu, ellapsed)')
    tmes, tmeb = time.clock(), time.time()
for it in range(ideb+ireb,ifin):
    # update modes
    if it > tims[1]:
        tits = [tit + 1 for tit in tits]
        tims = [tims[1], times[tits[1]]]
        wmodeb[0,...] = wmodeb[1,...]
        pmodeb[0,...] = pmodeb[1,...]
        norm_w[0,...] = norm_w[1,...]
        norm_p[0,...] = norm_p[1,...]
        Nsqb[0,...] = Nsqb[1,...]
        bbase[0,...] = bbase[1,...]
        (wmodeb[1,...], pmodeb[1,...]), (cmap, norm_w[1,...], norm_p[1,...]) \
                , (Nsqb[1,...], bbase[1,...]) = get_modes_interp_file( \
                        bastate_filename, tits[1], Nmodes \
                        , elem, coef, zw, zr, mask, doverb=True )
        ncw = Dataset(ncfilemode,'r+')
        ncw.variables['mode_speed'][tits[1],...] = cmap.T
        ncw.variables['w_norm'][tits[1],...] = norm_w[1,...].T
        ncw.variables['p_norm'][tits[1],...] = norm_p[1,...].T
        ncw.variables['time'][tits[1]] = tims[1]
        ncw.close()
        if doverb:
            print('({0},{1}), compute modes'.format(ipx,ipy), time.clock()-tmes\
                    , time.time()-tmeb)
            tmes, tmeb = time.clock(), time.time()
    simul.update(it,verbose=False)
    if dorep:
        simul.ncfile = simul.ncfile.replace(*repstr)    # var will read in different file
    wt = np.array([tims[1]-it, it-tims[0]])/(tims[1]-tims[0])   # weights
    bvf = (wt[:,None,None,None]*Nsqb).sum(axis=0)
    Nsqr = (bvf[...,1:] + bvf[...,-1:])/2.
    bbas = (wt[:,None,None,None]*bbase).sum(axis=0)
    wnorm = (wt[:,None,None,None]*norm_w).sum(axis=0)
    pnorm = (wt[:,None,None,None]*norm_p).sum(axis=0)
    wmod = (wt[:,None,None,None,None]*wmodeb).sum(axis=0)
    wmod = (wmod[:,:,1:,:] + wmod[:,:,:-1,:])/2.
    pmod = (wt[:,None,None,None,None]*pmodeb).sum(axis=0)
    zeta = var('zeta',simul,coord=coords).data[::nst,::nst]
    zz = comp_zlevs.zlev_rho(topo,zeta,simul.hc,simul.Cs_r)
    data = tools.vinterp(var('buoy',simul,coord=coords).data[::nst,::nst,:],zr,zz)\
            - bbas
    #res = np.trapz((data/Nsqb)[...,None]*rmodeb,zw[...,None],axis=-2)
    res = ( np.nansum((data*dzw)[...,None]*wmod,axis=-2) \
            + grav*(data[...,-1,None]/bvf[...,-1,None])*wmod[...,-1,:] ) / wnorm
    ncw = Dataset(ncfilename,'r+')
    ncw.variables['b_amp'][it-ideb,...] = res.T
    data = get_redpres(data,zr,0) + grav*zeta[...,None]
    res = np.nansum((data*dzw)[...,None]*pmod,axis=-2) / pnorm
    ncw.variables['p_amp'][it-ideb,...] = res.T
    data = var('w',simul,coord=coords).data[::nst,::nst,:]
    res = ( np.nansum((data*Nsqr*dzw)[...,None]*wmod,axis=-2) \
            + grav*data[...,-1,None]*wmod[...,-1,:] ) / wnorm
    ncw.variables['w_amp'][it-ideb,...] = res.T
    data = tools.u2rho(var('u',simul,coord=coordu).data)[1:-1:nst,::nst,:]
    res = np.nansum((data*dzw)[...,None]*pmod,axis=-2) / pnorm
    ncw.variables['u_amp'][it-ideb,...] = res.T
    data = tools.v2rho(var('v',simul,coord=coordv).data)[::nst,1:-1:nst,:]
    res = np.nansum((data*dzw)[...,None]*pmod,axis=-2) / pnorm
    ncw.variables['v_amp'][it-ideb,...] = res.T
    ncw.variables['time'][it-ideb] = simul.time
    ncw.variables['ocean_time'][it-ideb] = simul.oceantime/3600.
    ncw.close()
    if doverb:
        print('{0} -- {1:03} -- '.format((ipy,ipx),it),time.clock()-tmes,time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()
    

