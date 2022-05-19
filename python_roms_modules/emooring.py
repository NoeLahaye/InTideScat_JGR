from netCDF4 import Dataset
import numpy as np

class emooring(object):
    """ class containing all variables from a virtual mooring taken from a simulation 
    copy variables data ("ndplusArray" instance objects) and some meta_informations """
    
    def __init__(self,ncfile,itmin=None,itmax=None):
        nc = Dataset(ncfile,'r')
        ncvar = nc.variables
        
        if itmin is None:
            try:
                itmin = np.where(ncvar['time'][:].mask==False)[0][0]
            except:
                itmin = 0
        if itmax is None:
            try:
                itmax = np.where(ncvar['time'][:].mask==False)[0][-1] + 1
            except:
                itmax = nc.dimensions['time'].size
            
        # copy variables
        self.variables = {}
        for nam, val in ncvar.items():
            if 'time' in val.dimensions:# and val.ndim>1:
                inds = tuple([slice(itmin,itmax)]+[slice(0,ind) for ind in val.shape[1:]])
                setattr(self,nam,ndplusArray(val,inds))
            elif 'time' not in val.dimensions:
                setattr(self,nam,ndplusArray(val))
            self.variables[nam] = getattr(self,nam)
           
        # copy attributes (and create some)
        for att in ["hcoord_section","simul","date_beg","date_end"]:
            if att in nc.ncattrs():
                setattr(self,att,nc.getncattr(att))
        self.nt = itmax - itmin
        self.nx = nc.dimensions['xi_rho'].size
        self.ny = nc.dimensions['eta_rho'].size
        nc.close()
        
#class var_from_netCDF(np.ndarray):
    #""" class to store data from netCDF file with attributes """
    #def __init__(self,ncvar,indices=None):
        #if indices is None:
            #self = ncvar[:]
        #else:
            #self = ncvar[indices]
        #self.dims = ncvar.dimensions
        #for att in self.ncattrs():
            #setattr(self,att,ncvar.getncattr(att))
            
class ndplusArray(np.ndarray):
    """ subclass of ndarray for taking netCDF variables with variable attributes 
    see https://docs.scipy.org/doc/numpy/user/basics.subclassing.html """
    def __new__(cls,ncvar,indices=None):
        if indices is None:
            indices = tuple([slice(0,ind) for ind in ncvar.shape])
        if isinstance(ncvar[indices],np.ma.masked_array):
            data = ncvar[indices].astype(float).filled(np.nan)
        else:
            data = ncvar[indices]
        obj = np.asarray(data).view(cls)
        attrs = {key:ncvar.getncattr(key) for key in ncvar.ncattrs()}
        obj.ncattrs = attrs
        for key,val in attrs.items():
            setattr(obj,key,val)
        return obj

    def __array_finalize__(self,obj):
        if obj is None: return
        self.ncattrs = getattr(obj,"ncattrs",None)

