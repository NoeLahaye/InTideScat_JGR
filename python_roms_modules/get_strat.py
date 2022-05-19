from __future__ import print_function, division
from scipy import interpolate
from netCDF4 import Dataset

''' read vertical profiles of temperature and salinity in a given netcdf name, 
interpolate in the vertical on a 3D space grid --  currently uses linear interpolation 
N. Lahaye April 2017 for LUCKYTO    '''

def interpolator(xx,yy):        # wrapper
    fillval = yy[-1]     # this is gros bricolage
    return interpolate.interp1d(xx,yy,kind='linear',axis=-1,bounds_error=False,fill_value=fillval)
# had to set bound error and fill value because extrapolate not available on Curie 
# (need version >= 0.17.0)

def get_strat(strname,zz):
    ncvar = Dataset(strname,'r').variables      # import 1D fields (homogeneous stratif)
    zlev = ncvar['z_rho'][:]        # vertical levels of rho points
    temp = ncvar['temp'][:]         # temperature vertical profile
    salt = ncvar['salt'][:]         # salinity vertical profile

    ftemp = interpolator(zlev,temp)     # build interpolating function
    fsalt = interpolator(zlev,salt)     # z-axis is the last one, whatever the shape

    return ftemp(zz), fsalt(zz)

def interp_ts_pchip(zlev,temp,salt,zz):
    #ncvar = Dataset(strname,'r').variables
    #zlev = ncvar['depth'][:]
    #temp = ncvar['temp'][:]
    #salt = ncvar['salt'][:]

    ftemp = interpolate.PchipInterpolator(zlev,temp,axis=0,extrapolate=True)
    fsalt = interpolate.PchipInterpolator(zlev,salt,axis=0,extrapolate=True)

    return ftemp(zz), fsalt(zz) 
    
