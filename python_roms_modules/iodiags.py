from __future__ import print_function
import os
from netCDF4 import Dataset
import numpy as np
# TODO fix vname argument passage conflicting with kwargs

dim0, dim3, dim2, dim1 = ['time', ['zlev'], ['eta_rho','eta_v'], ['xi_rho','xi_u']]

class wr_slab(object):
    # write y*x slabs at given time (snapshots) from 2D or 3D variable
    # most of the instructions are from R_netcdf module
    # input is a (x,y) axis-ordered array or R_var instance variable
    # store order is (t,y,x) 
    def __init__(self,fname,simul,var,it=None,zlev=0.,iz=0,**kwargs):
    ### in kwargs: dico of attributes ncatts, varattc, vname, coord
    ### this version does not include writing subdomains
        if 'netcdf_format' in kwargs:
            netcdf_format = kwargs['netcdf_format']
        else:
            netcdf_format = 'NETCDF4'
        if not os.path.isfile(fname): 
            nc = _createfil(fname,simul,netcdf_format=netcdf_format)
        else:
            nc = Dataset(fname,'a')
        try: 
            vname = var.name
            data = var.data.T
        except (AttributeError, TypeError): 
            if 'vname' in kwargs: vname = kwargs.pop('vname')
            else: vname = 'var'
            data = var.T
        if vname not in nc.variables:
            self.createvar(nc,var,vname,**kwargs)
        if it == None: it = nc.dimensions[dim0].__len__()
        if data.ndim > 2:
            nc.variables[vname][it,:,:] = data[iz,:,:]
        else:
            nc.variables[vname][it,:,:] = data
        nc.variables['ocean_time'][it] = simul.oceantime
        nc.variables['time'][it] = simul.time
        nc.close()

    def createvar(self,nc,var,vname,**kwargs):   # this looks like a class method
    # note: would be better if we had shape and dims defined in var class instance
    # also attributes and inheritance of getncattr
        try:
            jmin = var.jmin; imin = var.imin
            dims = (dim2[jmin],dim1[imin])      # order y, x
        except:
            try:
                dims = kwargs['dims']       # warning need same order as shape
                #jmin = dim2.index(dims[0])
            except:
                raise Exception('Unable to find dimensions for variable '+vname)
        try: 
            lasize = var.shape[::-1]        # shape of transposed array
        except (AttributeError, TypeError): 
            lasize = var.data.shape[::-1]
        for (siz,dim) in zip(lasize,dims):
            if dim not in nc.dimensions:
               nc.createDimension(dim,siz)
        if 'chunksizes' in kwargs: chsize = kwargs['chunksizes']
        else: chsize = None
        ncvar = nc.createVariable(vname,'f',(dim0,)+dims,chunksizes=chsize)
        try: 
            ncvar.setncattr('long_name',var.long_name)
        except: 
            try: 
                ncvar.setncattr('long_name',var.longname)
            except: pass
        try: 
            ncvar.setncattr('units',var.units)
        except: 
            try: 
                ncvar.setncattr('units',var.unit)
            except: pass    
        try: 
            ncvar.setncattr('coord',var.coord[:4])
        except: pass    
        if 'varatts' in kwargs:
            ncvar.setncatts(kwargs['varatts'])
        return ncvar

def _createfil(fname,simul,netcdf_format='NETCDF4',tsize=None,**kwargs):
    nc = Dataset(fname,'w',format=netcdf_format)
    nc.createDimension(dim0,tsize)       # create time dimension (common to all)
    nc.createVariable(dim0,'i4',(dim0))  # associated variable (for indices)
    nc.createVariable('ocean_time','f',(dim0))  # associated ocean time
    nc.setncattr('simul',simul.simul)
    if 'ncatts' in kwargs:                # write attributes (ncattr is a dico)
        nc.setncatts(kwargs['ncatts'])
    return nc                           # N.B.: this could be a class method
    
class wr_tseries(object):
    # writeatime series of x*y slabs (3D blocks, y*x*t) in 4D var (z*...)
    # most of the instructions are from R_netcdf module
    # input is a (y,x,t) axis-ordered array or R_var instance variable
    # store order is (z,y,x,t), every dimensions are fix w/ auto chunks in y,x,t
    def __init__(self,fname,simul,var,zlev=0.,iz=0,**kwargs):
    ### in kwargs: dico of attributes ncatts, varattc, vname, coord, 
    ### this version does not include writing subdomains
        if 'netcdf_format' in kwargs:
            netcdf_format = kwargs['netcdf_format']
        else:
            netcdf_format = 'NETCDF4'
        if not os.path.isfile(fname): 
            nc = _createfil(fname,simul,netcdf_format=netcdf_format,tsize=var.shape[-1])
        else:
            nc = Dataset(fname,'a')
        if 'vname' in kwargs: 
            vname = kwargs['vname']
        else:
            try: 
                vname = var.name
            except (AttributeError, TypeError): 
                vname = 'var'
            kwargs['vname'] = vname
        if isinstance(var,np.ndarray):
            data = var
        else:
            data = var.data
        if vname not in nc.variables:
            self.createvar(nc,var,**kwargs)
        print('netcdf variable dims & shape:',nc.variables[vname].dimensions,nc.variables[vname].shape)
        print('data dims and shape',kwargs['dims'],data.shape)
        if 'inds' in kwargs:  
            vecy = range(kwargs['inds'][0],kwargs['inds'][1])
            vecx = range(kwargs['inds'][2],kwargs['inds'][3])
            print(vecy[0],vecy[-1],vecx[0],vecx[-1])
            nc.variables[vname][iz,vecy,vecx,:] = data
        else:
            vecy = range(data.shape[0])
            vecx = range(data.shape[1])
            print(vecy[0],vecy[-1],vecx[0],vecx[-1])
            nc.variables[vname][iz,vecy,vecx,:] = data
        nc.close()

    def createvar(self,nc,var,**kwargs):   # this looks like a class method
    # note: would be better if we had shape and dims defined in var class instance
    # also attributes and inheritance of getncattr
        vname = kwargs['vname']     # I suppressed explicit vname and put it in kwargs
        try:
            dims = (dim3,dim2[var.jmin],dim1[var.imin])      # order y, x
        except:
            try:
                dims = kwargs['dims']
            except:
                raise Exception('Unable to find dimensions for variable '+vname)
        if 'coord' in kwargs:
            coord = kwargs['coord']
            lasize = (coord[1]-coord[0],coord[3]-coord[2])
        else:
            try: 
                lasize = var.shape
            except (AttributeError, TypeError): 
                lasize = var.data.shape
        if len(lasize)<4: lasize=(None,)+lasize     # hand-made detect outer dim
        elif 1 in lasize:
            ione = lasize.index(1)[0]
            lasize = lasize[:ione]+(None,)+lasize[ione+1:]
        print(lasize,dims)
        for (ii,dim) in zip(range(4),dims):        
            if dim not in nc.dimensions:
                nc.createDimension(dim,lasize[ii])
                nc.createVariable(dim,'f',(dim))
        if 'chunksizes' in kwargs: chsize = kwargs['chunksizes']
        else: chsize = None
        ncvar = nc.createVariable(vname,'f',dims,chunksizes=chsize)
        try: 
            ncvar.setncattr('long_name',var.long_name)
        except: 
            try: 
                ncvar.setncattr('long_name',var.longname)
            except: pass
        try: 
            ncvar.setncattr('units',var.units)
        except: 
            try: 
                ncvar.setncattr('units',var.unit)
            except: pass    
        try: 
            ncvar.setncattr('coord',var.coord)
        except: pass
        if 'varatts' in kwargs:
            ncvar.setncatts(kwargs['varatts'])
        return ncvar


class wr_vslab(object):
    # write vertical slab (fields on sigma levels interpolated on a 1D horizontal grid)
    # assume that data is a 2D array (l,z)
    def __init__(self,fname,simul,var,**kwargs):
        if "netcdf_format" in kwargs:
            netcdf_format = kwargs['netcdf_format']
        else:
            netcdf_format = "NETCDF4"
        if not os.path.isfile(fname): 
            nc = _createfil(fname,simul,netcdf_format=netcdf_format,**kwargs)
            if 'ncatts' in kwargs:
                nc.setncatts(kwargs['ncatts'])
        else:
            nc = Dataset(fname,'a')
        if 'vname' in kwargs: vname = kwargs.pop('vname')
        else: vname = 'var'
        data = var
        if vname not in nc.variables:
            ncvar = self.createvar(nc,var,vname,**kwargs)
        else:
            ncvar = nc.variables[vname]
        if dim0 not in ncvar.dimensions:  # for coordinate variable
            nc.variables[vname][:] = data
        else:
            try:
                it = kwargs['it']
            except:
                it = nc.dimensions[dim0].__len__()
            nc.variables[vname][it,:,:] = data
            nc.variables['time'][it] = simul.time
            nc.variables['ocean_time'][it] = simul.oceantime
        nc.close()

    def createvar(self,nc,var,vname,**kwargs):
        try:
            dims = kwargs['dims']       # warning need same order as shape
        except:
            raise Exception('Unable to find dimensions for variable '+vname)
        lasize = var.shape
        for (si,dim) in zip(lasize,dims):
            if dim not in nc.dimensions:
                nc.createDimension(dim,si)
                #if dim not in ['lslab','time']:
                    #nc.createVariable(dim,'f',(dim,))
                    #nc.variables[dim][:] = range(si)
        try:
            ncvar = nc.createVariable(vname,var.dtype,dims)
        except:
            ncvar = nc.createVariable(vname,'f',dims)
        if 'varatts' in kwargs:
            ncvar.setncatts(kwargs['varatts'])
        return ncvar


class pathdef(object):

    def __init__(self,dirname='data',doappend='True'):
        self.dirp = dirname+'/'
        self.cncr_dir(dirname)
        self.doapp = doappend

    def cncr_dir(self,dirname):
    #''' Check and Create directory
    #   check if directory "dirname" exists, otherwise create it, including all 
    #   intermediate-level directories needed '''
        if not os.path.isdir(dirname):
            print(dirname,'directory does not exist: creating it')
            os.makedirs(dirname)

    def tree_dir(self,sdirname):
    # add subdirectories to an existing directory
    # create folder(s) if needed and update self.dirp name
        self.dirp = self.dirp+sdirname+'/'
        if not os.path.isdir(self.dirp):
            os.makedirs(self.dirp)

    def ficdir(self,name=''):
    #''' return file name with path string from name of file '''
        return self.dirp+name

    def cnch_fil(self,ficname,**kwargs):
    #''' Check and Change filename
    #   check if filename in self.dirp directory exists, change name if so, and return '''
        if 'verbose' in kwargs: verb = kwargs['verbose'] 
        else: verb = True
        if not self.doapp and os.path.isfile(self.ficdir(ficname)):
            oldname = ficname
            iext = ficname.rfind('.')       # identify extension
            ifil = 1
            ficname = ficname[:iext]+str(ifil)+ficname[iext:]
            print(ficname)
            while os.path.isfile(self.ficdir(ficname)): 
                ifil += 1
                ficname = ficname[:iext]+str(ifil)+ficname[iext+1:]
                print(ficname)
            if verb: 
                print(self.ficdir(oldname),'already exists. New file name is:',ficname)
        return ficname

