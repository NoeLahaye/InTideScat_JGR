from __future__ import print_function, division
import time
import numpy as np
from netCDF4 import Dataset
import scipy.interpolate as itp
import R_tools_fort as toolsf
from dynmodes_fs import dynmodes_w

grav = 9.81

def get_modes_interp_file(filename,it,Nmodes,elems,coef,zwb,zrb,mask=None,doverb=True):
    """ input:
    filename
    it
    nmodes
    elems
    coef
    zwb
    zrb
    mask=None
    doverb=True
    returns: (wmodeb, pmodeb), (cmap, norm_w, norm_p), (Nsqb, bbase)
    order: ((t,) z, y, x) in bastate file, returns (x, y, z, M)
    """
    nc = Dataset(filename)
    ny, nx = nc.variables['h'].shape
    ii,jj = np.unravel_index(elems, (nx,ny))    # indices in full grid
    i1, i2 = ii.min (), ii.max()+1
    j1, j2 = jj.min(), jj.max()+1
    topo = nc.variables['h'][j1:j2,i1:i2].T
    nx, ny = topo.shape
    hc, Cs_r, Cs_w = nc.hc, nc.Cs_r, nc.Cs_w
    zr, zw = toolsf.zlevs(topo,np.zeros((nx,ny)),hc,Cs_r,Cs_w)
    del topo, hc, Cs_r, Cs_w
    bbase = nc.variables['buoy_lowf'][it,:,j1:j2,i1:i2]
    Nsqb = nc.variables['bvf_lowf'][it,:,j1:j2,i1:i2]
    nc.close()
    elems = np.ravel_multi_index((ii-i1,jj-j1),(nx,ny)) # indices in sub grid
    # z-interpolation
    bvf_itp, buo_itp = [], []
    if doverb:
        tmes, tmeb = time.clock(), time.time()
    for ii in range(nx):
        for jj in range(ny):
            bvf_itp.append(itp.pchip(zw[ii,jj,:],np.pad(Nsqb[:,jj,ii],1,"edge")))
            #buo_itp.append(itp.pchip(zr[ii,jj,:],bbase[:,jj,ii]))
            buo_itp.append(itp.interp1d(zr[ii,jj,:], bbase[:,jj,ii], kind="linear" \
                            , fill_value="extrapolate")) # no crazy extrapolated values
    bvf_itp = np.array(bvf_itp)
    buo_itp = np.array(buo_itp)
    del Nsqb, bbase, zr, zw
    if doverb:
        print("interpolated",time.clock()-tmes,time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()

    Nx, Ny, Nzw = zwb.shape
    Nzr = zrb.shape[-1]
    if mask is None:
        mask = np.ones((Nx,Ny))
    cmap = np.full((Nx,Ny,Nmodes),np.nan,dtype=np.float32)
    wmodeb = np.full((Nx,Ny,Nzw,Nmodes),np.nan)
    Nsqb = np.full((Nx,Ny,Nzw),np.nan,dtype=np.float32)
    bbase = np.full((Nx,Ny,Nzr),np.nan,dtype=np.float32)

    nmodes = Nmodes
    for ix,iy in zip(*np.where(np.isfinite(mask))):
        valitp = np.array([bvf_itp[kk](zwb[ix,iy,1:-1]) for kk in elems[ix,iy,:]])
        bvf = np.pad((coef[ix,iy,:,None]*valitp).sum(axis=0),1,'edge')
        Nsqb[ix,iy,:] = bvf
        try:
            wmode, ce, _ = dynmodes_w(bvf,zwb[ix,iy,:],Nmodes,fs=True)
            nmodes = wmode.shape[-1]
            if nmodes < Nmodes:
                nmodes = wmode.shape[-1]
                print(ix,iy,"found only",nmodes,"modes")
                wmodeb[ix,iy,1:-1,-3:] = valitp.T
                wmodeb[ix,iy,0,-3:] = coef[ix,iy,:]
                wmodeb[ix,iy,-1,-3:] = elems[ix,iy,:]
            cmap[ix,iy,:nmodes] = ce[:nmodes]
            wmodeb[ix,iy,:,:nmodes] = wmode[:,:nmodes]
        except:
            print(ix,iy,"ca a foir'e")
        valitp = np.array([buo_itp[kk](zrb[ix,iy,:]) for kk in elems[ix,iy,:]])
        bbase[ix,iy,:] = (coef[ix,iy,:,None]*valitp).sum(axis=0)
    if doverb:
        print('ended computing {0} modes \n\t'.format(Nmodes)\
            , "Timing:",time.clock()-tmes, time.time()-tmeb)
        tmes, tmeb = time.clock(), time.time()
    pmodeb  = np.diff(wmodeb,axis=-2)/np.diff(zwb,axis=-1)[...,None]
    pmodeb = pmodeb/np.abs(pmodeb).max(axis=-2)[...,None,:]
    norm_w = np.trapz(Nsqb[...,None]*wmodeb**2,zwb[...,None],axis=-2) + grav*wmodeb[...,-1,:]**2
    zwb = np.diff(zwb,axis=-1)  # zwb is now dz
    norm_p = np.nansum(zwb[...,None]*pmodeb**2,axis=-2)

    return (wmodeb, pmodeb), (cmap, norm_w, norm_p), (Nsqb, bbase)

