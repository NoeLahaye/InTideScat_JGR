from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset, MFDataset
from scipy.spatial import Delaunay
import scipy.interpolate as itp
from matplotlib import pyplot as plt
import time
from R_smooth import get_tri_coef
import R_tools_fort as toolsf
plt.ion()

filename = "/net/ruchba/local/tmp/2/lahaye/luckyt_tseries_lf/luckyt_subsamp_bvf.nc"
grdfile = "/net/ruchba/local/tmp/2/lahaye/prep_LUCKYTO/lucky_corgrd.nc"
tserfil = "/net/krypton/data0/project/vortex/lahaye/luckyt_tseries/luckyt_tseries_2Dvars.nc"

whatdim = 4     # do 2D, 3D (z) or 4D (t)

if whatdim == 2:
    nc = Dataset(filename)
    lon = nc.variables['lon_rho'][:]
    lat = nc.variables['lat_rho'][:]
    data = nc.variables['bvf_avg'][-10,:,:]
    nc.close()
    
    nc = Dataset(filename.replace('subsamp','lowf'))
    xx = nc.variables['lon_rho'][:]
    yy = nc.variables['lat_rho'][:]
    doto = nc.variables['bvf_avg'][-10,:,:]
    nc.close()
    
    # bilinear interpolation using delaunay triangulation
    elem, coef = get_tri_coef(lon,lat,xx,yy)
    dati = (coef*data.ravel()[elem]).sum(axis=-1)
    
    # cubic based on Delaunay
    #res = Delaunay(np.c_[lon.ravel(),lat.ravel()])
    #ras = itp.CloughTocher2DInterpolator(res,data.ravel())

elif whatdim == 3:
    imin, imax = 200, 400
    jmin, jmax = 800, 1000

    nc = Dataset(filename.replace('subsamp','lowf'))
    xi = nc.variables['xi_rho'][:]
    indx, = np.where((xi>=imin) & (xi<=imax))
    eta = nc.variables['eta_rho'][:]
    indy, = np.where((eta>=jmin) & (eta<=jmax))
    xx = nc.variables['lon_rho'][indy,indx].T
    yy = nc.variables['lat_rho'][indy,indx].T
    doto = nc.variables['bvf_avg'][:,indy,indx]
    Nx, Ny = xx.shape
    xo, eto = xi[indx], eta[indy]
    nc.close()
    # load topography to compute zlevs
    nc = Dataset(grdfile)
    topo = nc.variables['h'][eta[indy],xi[indx]].T
    nc.close()
    nc = Dataset(tserfil)
    hc = nc.hc
    Cs_r = nc.Cs_r
    Cs_w = nc.Cs_w
    nc.close()
    zr, zw = toolsf.zlevs(topo,np.zeros(topo.shape),hc,Cs_r,Cs_w)

    nc = Dataset(filename)
    xi = nc.variables['xi_rho'][:]
    i1 = np.where(xi>imin)[0][0]-1
    i2 = np.where(xi<imax)[0][-1]+2
    eta = nc.variables['eta_rho'][:]
    j1 = np.where(eta>jmin)[0][0]-1
    j2 = np.where(eta<jmax)[0][-1]+2
    lon = nc.variables['lon_rho'][j1:j2,i1:i2].T
    lat = nc.variables['lat_rho'][j1:j2,i1:i2].T
    data = nc.variables['bvf_avg'][:,j1:j2,i1:i2]
    nc.close()
    nc = Dataset(grdfile)
    h = nc.variables['h'][eta[j1:j2],xi[i1:i2]].T
    nc.close()
    _, zz = toolsf.zlevs(h,np.zeros(h.shape),hc,Cs_r,Cs_w)
    nx, ny = h.shape
    xi, eta = xi[i1:i2], eta[j1:j2]
    zref= zz[0,0,1:-1]/h[0,0]   # for vertical integration
    
    # bilinear interpolation using delaunay triangulation
    tmes, tmeb = time.clock(), time.time()
    elem, coef = get_tri_coef(lon,lat,xx,yy)
    print("triangulation",time.clock()-tmes,time.time()-tmeb)

    # z-interpolation
    finterp = []
    tmes, tmeb = time.clock(), time.time()
    for ii in range(nx):
        for jj in range(ny):
            #finterp.append(itp.CubicSpline(zz[ii,jj,:],data[:,jj,ii]))
            finterp.append(itp.pchip(zz[ii,jj,:],np.pad(data[:,jj,ii],1,"edge")))
    finterp = np.array(finterp)
    print("interpolated",time.clock()-tmes,time.time()-tmeb)
    
    tmes, tmeb = time.clock(), time.time()
    datint = np.zeros(doto.shape)
    for ii in range(Nx):
        for jj in range(Ny):
            valitp = np.array([finterp[kk](zw[ii,jj,1:-1]) for kk in elem[ii,jj,:]])
            datint[:,jj,ii] = (coef[ii,jj,:,None]*valitp).sum(axis=0)

    print("reconstructed",time.clock()-tmes,time.time()-tmeb)

    # compute RMS error
    rms_err = np.sqrt(np.trapz(((datint - doto)**2),zref,axis=0)\
                    /np.trapz((doto**2),zref,axis=0))
    plt.imshow(np.log10(rms_err),vmin=-6,vmax=0); plt.colorbar()

    iy, ix = np.unravel_index(np.argmax(rms_err),rms_err.shape)
    plt.figure()
    plt.plot(doto[:,iy,ix],zw[ix,iy,1:-1],'.-k')
    plt.plot(datint[:,iy,ix],zw[ix,iy,1:-1],'.-b')

    iz = 40
    fig, axs = plt.figure(figsize=(10,5))
    axs[0].pcolormesh(doto[iz,:,:],vmin=datint[iz,:,:].min(),vmax=datint[iz,:,:].max())
    axs[1].pcolormesh(datint[iz,:,:],vmin=datint[iz,:,:].min(),vmax=datint[iz,:,:].max())

#####  ---  Four dimensional
elif whatdim == 4:
    imin, imax = 200, 400
    jmin, jmax = 800, 1000
    it = 10     # time index (original or subsampled time series)

    nc = Dataset(filename.replace('subsamp','lowf'))
    xi = nc.variables['xi_rho'][:]
    indx, = np.where((xi>=imin) & (xi<=imax))
    eta = nc.variables['eta_rho'][:]
    indy, = np.where((eta>=jmin) & (eta<=jmax))
    xx = nc.variables['lon_rho'][indy,indx].T
    yy = nc.variables['lat_rho'][indy,indx].T
    Nx, Ny = xx.shape
    xo, eto = xi[indx], eta[indy]
    doto = nc.variables['bvf_lowf'][:,indy,indx,it]
    nc.close()
    # load topography to compute zlevs
    nc = Dataset(grdfile)
    topo = nc.variables['h'][eta[indy],xi[indx]].T
    nc.close()
    nc = Dataset(tserfil)
    hc = nc.hc
    Cs_r = nc.Cs_r
    Cs_w = nc.Cs_w
    timef = nc.variables['time'][:]
    nc.close()
    zr, zw = toolsf.zlevs(topo,np.zeros(topo.shape),hc,Cs_r,Cs_w)

    nc = Dataset(filename)
    xi = nc.variables['xi_rho'][:]
    i1 = np.where(xi>imin)[0][0]-1
    i2 = np.where(xi<imax)[0][-1]+2
    eta = nc.variables['eta_rho'][:]
    j1 = np.where(eta>jmin)[0][0]-1
    j2 = np.where(eta<jmax)[0][-1]+2
    lon = nc.variables['lon_rho'][j1:j2,i1:i2].T
    lat = nc.variables['lat_rho'][j1:j2,i1:i2].T
    data = nc.variables['bvf_lowf'][:,:,j1:j2,i1:i2]    # t, z
    times = nc.variables['time'][:]
    nc.close()
    nc = Dataset(grdfile)
    h = nc.variables['h'][eta[j1:j2],xi[i1:i2]].T
    nc.close()
    _, zz = toolsf.zlevs(h,np.zeros(h.shape),hc,Cs_r,Cs_w)
    nx, ny = h.shape
    xi, eta = xi[i1:i2], eta[j1:j2]
    zref= zz[0,0,1:-1]/h[0,0]   # for vertical integration
    print('just finished loading stuffs')
    
    # bilinear interpolation using delaunay triangulation
    tmes, tmeb = time.clock(), time.time()
    elem, coef = get_tri_coef(lon,lat,xx,yy)
    print("triangulation",time.clock()-tmes,time.time()-tmeb)

    # z-interpolation
    finterp = []
    tmes, tmeb = time.clock(), time.time()
    for ii in range(nx):
        for jj in range(ny):
            finterp.append(itp.RectBivariateSpline(\
                    times,zz[ii,jj,:],np.pad(data[:,:,jj,ii],((0,0),(1,1)),'edge')))
    finterp = np.array(finterp)
    print("interpolated",time.clock()-tmes,time.time()-tmeb)
    
    tmes, tmeb = time.clock(), time.time()
    datint = np.zeros(doto.shape)
    for ii in range(Nx):
        for jj in range(Ny):
            valitp = np.array([finterp[kk](times[it],zw[ii,jj,1:-1]).squeeze() \
                                for kk in elem[ii,jj,:]])
            datint[:,jj,ii] = (coef[ii,jj,:,None]*valitp).sum(axis=0)

    print("reconstructed",time.clock()-tmes,time.time()-tmeb)

    # compute RMS error
    rms_err = np.sqrt(np.trapz(((datint - doto)**2),zref,axis=0) \
                    /np.trapz((doto**2),zref,axis=0))
    plt.imshow(np.log10(rms_err),vmin=-6,vmax=0); plt.colorbar()

    iy, ix = np.unravel_index(np.argmax(rms_err),rms_err.shape)
    plt.figure()
    plt.plot(doto[:,iy,ix],zw[ix,iy,1:-1],'.-k')
    plt.plot(datint[:,iy,ix],zw[ix,iy,1:-1],'.-b')

    iz = 40
    fig, axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))
    axs[0].pcolormesh(doto[iz,:,:],vmin=datint[iz,:,:].min(),vmax=datint[iz,:,:].max())
    axs[1].pcolormesh(datint[iz,:,:],vmin=datint[iz,:,:].min(),vmax=datint[iz,:,:].max())

    # reconstruct time series
    ii, jj = ix, iy
    valitp = np.array([finterp[kk](timef,zw[ii,jj,1:-1]).squeeze() \
                        for kk in elem[ii,jj,:]])
    datent = (coef[ii,jj,:,None,None]*valitp).sum(axis=0)
    ii, jj = 2, 5
    valitp = np.array([finterp[kk](timef,zw[ii,jj,1:-1]).squeeze() \
                        for kk in elem[ii,jj,:]])
    datant = (coef[ii,jj,:,None,None]*valitp).sum(axis=0)

    plt.figure()
    plt.plot(times,data[:,iz,2,1],'ok',timef,datant[:,iz],'-k')
    inds = [np.where(timef==ii)[0][0] for ii in times] 
    plt.plot(timef,datent[:,iz],'-b',times,datent[inds,iz],'ob')
