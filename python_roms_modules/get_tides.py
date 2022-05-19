from __future__ import print_function, division
from netCDF4 import Dataset
import numpy as np

''' from doc in croco_tools Tides doc (Zhighang Xu, Nov. 2000)
read tidal data and rebuild corresponding elevation, x- and y- velocity fields
at a given tide. Each tidal component is kept separated along an extra dimension 
time must be in hour. N.B.: ubar, vbar are pure azimutal and meridional components
N. Lahaye April 2017 for LUCKYTO    '''

def get_alltides(frcname,time=0.,doverb=False):
### zeta, ubar, vbar = get_tides(frcname,doverb=False)
### zeta, ubar, vbar have same shape as tidal components stored in "frcname"
    ncfrc = Dataset(frcname,'r')
    if doverb:
        print('getting tidal signal from file',frcname)
    Ntides = len(ncfrc.dimensions['tide_period'])
    tide_period = ncfrc.variables['tide_period'][:] 
    if doverb:
        print(Ntides,'components with period (h):',tide_period)
    omega = 2*np.pi/tide_period[:,None,None]
### Sea-surface elevation
    zeta_amp = ncfrc.variables['tide_Eamp'][:]
    zeta_phi = ncfrc.variables['tide_Ephase'][:]*np.pi/180.
    zeta = zeta_amp*np.cos(omega*time - zeta_phi)
    if doverb:
        print('computed surface elevation, shapes:',zeta_amp.shape,zeta_phi.shape,zeta.shape)
    del zeta_amp, zeta_phi, tide_period

    ### Current
    Cmin = ncfrc.variables['tide_Cmin'][:]
    Cmax = ncfrc.variables['tide_Cmax'][:]
    Cphi = ncfrc.variables['tide_Cphase'][:]*np.pi/180.
    Cang = ncfrc.variables['tide_Cangle'][:]*np.pi/180.
    ncfrc.close()
    Wp = (Cmax + Cmin)/2.
    Wm = (Cmax - Cmin)/2.
    Thetap = Cang - Cphi
    Thetam = Cang + Cphi
    del Cmin, Cmax, Cphi, Cang

    #ww = Wp*np.exp(1j*(omega*time - Thetap)) + Wm*np.exp(1j*(Thetam - omega*time))
    ww = Wp*np.exp(1j*(omega*time + Thetap)) + Wm*np.exp(1j*(Thetam - omega*time))
    ubar = ww.real
    vbar = ww.imag

    return zeta, ubar, vbar

def get_sumtides(frcname,time=0.,itides=None,doverb=False):
### zeta, ubar, vbar = get_tides(frcname,doverb=False)
### zeta, ubar, vbar have same shape as tidal components stored in "frcname"
    ncfrc = Dataset(frcname,'r')
    ncvar = ncfrc.variables
    if doverb:
        print('getting tidal signal from file',frcname)
    Ntides = len(ncfrc.dimensions['tide_period'])
    if itides is None:
        itides = range(Ntides)
    tide_period = ncfrc.variables['tide_period'][itides] 
    zeta = np.zeros(ncfrc.variables['tide_Eamp'].shape[1:])
    ubar = np.copy(zeta); vbar = np.copy(zeta)
    
    if doverb:
        print(Ntides,'components with period (h):',tide_period)
        print('using components',itides)
    omega = 2*np.pi/tide_period[:,None,None]
    for ip in itides:
        zeta += ncvar['tide_Eamp'][ip,...]*np.cos(omega[ip,...]*time \
                - ncvar['tide_Ephase'][ip,...]*np.pi/180.)
        Cmin = ncfrc.variables['tide_Cmin'][ip,...]
        Cmax = ncfrc.variables['tide_Cmax'][ip,...]
        Cphi = ncfrc.variables['tide_Cphase'][ip,...]*np.pi/180.
        Cang = ncfrc.variables['tide_Cangle'][ip,...]*np.pi/180.
        Wp = (Cmax + Cmin)/2.
        Wm = (Cmax - Cmin)/2.
        Thetap = Cang - Cphi
        Thetam = Cang + Cphi
        ww = Wp*np.exp(1j*(omega[ip,...]*time + Thetap)) + Wm*np.exp(1j*(Thetam - omega[ip,...]*time))
        ubar += ww.real
        vbar += ww.imag

    return zeta, ubar, vbar

def get_sumtides_onepoint(frcname,times=np.array(0.),ind=[0,0],doverb=False):
### zeta, ubar, vbar = get_tides(frcname,doverb=False)
### zeta, ubar, vbar are time series at single point
    ix, iy = ind
    ncfrc = Dataset(frcname,'r')
    ncvar = ncfrc.variables
    if doverb:
        print('getting tidal signal from file',frcname)
    Ntides = len(ncfrc.dimensions['tide_period'])
    tide_period = ncfrc.variables['tide_period'][:] 
    zeta = np.zeros(times.shape) 
    ubar = np.copy(zeta); vbar = np.copy(zeta)
    
    if doverb:
        print(Ntides,'components with period (h):',tide_period)
    omega = 2*np.pi/tide_period
    zamp = ncvar['tide_Eamp'][:,iy,ix]
    zphi = ncvar['tide_Ephase'][:,iy,ix]*np.pi/180.
    Cmin = ncfrc.variables['tide_Cmin'][:,iy,ix]
    Cmax = ncfrc.variables['tide_Cmax'][:,iy,ix]
    Cphi = ncfrc.variables['tide_Cphase'][:,iy,ix]*np.pi/180.
    Cang = ncfrc.variables['tide_Cangle'][:,iy,ix]*np.pi/180.
    Wp = (Cmax + Cmin)/2.
    Wm = (Cmax - Cmin)/2.
    Thetap = Cang - Cphi
    Thetam = Cang + Cphi
    for ip in range(Ntides):
        zeta += zamp[ip]*np.cos(omega[ip]*times - zphi[ip])
        ww = Wp[ip]*np.exp(1j*(omega[ip]*times + Thetap[ip])) \
            + Wm[ip]*np.exp(1j*(Thetam[ip] - omega[ip]*times))
        ubar += ww.real
        vbar += ww.imag

    return zeta, ubar, vbar

def get_sumtides_zeta(frcname,time=0.,tramp=None,itides=None,doverb=False):
### zeta, zetat = get_tides(frcname,time,doverb=False)
### zeta, zetat have same shape as tidal components stored in "frcname"
    if tramp is None:
        ramp = 1.   
        dramp = 0.
    else:
        ramp = np.tanh(time/tramp)  # tramp in hours
        dramp = 1./tramp/np.cosh(time/tramp)**2
    ncfrc = Dataset(frcname,'r')
    ncvar = ncfrc.variables
    if doverb:
        print('getting tidal signal from file',frcname)
    Ntides = len(ncfrc.dimensions['tide_period'])
    if itides is None:
        itides = range(Ntides)
    tide_period = ncfrc.variables['tide_period'][itides] 
    zeta = np.zeros(ncfrc.variables['tide_Eamp'].shape[1:]); 
    zetat = np.copy(zeta)
    if doverb:
        print(Ntides,'components with period (h):',tide_period)
        print('using components',itides)
    omega = 2*np.pi/tide_period[:,None,None]
    for ip in itides:
        amp = ncvar['tide_Eamp'][ip,...]
        phi = ncvar['tide_Ephase'][ip,...]*np.pi/180.
        zeta += ramp*amp*np.cos(omega[ip,...]*time - phi)
        zetat += -ramp*amp*omega[ip,...]*np.sin(omega[ip,...]*time - phi)
        if tramp is not None:
            zetat += dramp*amp*np.cos(omega[ip,...]*time - phi)

    return zeta, zetat
