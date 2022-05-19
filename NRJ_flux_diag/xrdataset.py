import xarray as xr

def xrdataset(filenames,aggdims,no_mf=True):
    """ filenames: shape 2 list [dim y [dim x]] ; aggdims = [dimy,dimx] """
    ncs = []
    for names in filenames:
        if no_mf:
            ncc = [xr.open_dataset(name) for name in names]
            ncs.append( \
                    xr.concat(ncc, dim=aggdims[-1], data_vars='minimal') )
        else:
            ncs.append( xr.open_mfdataset(names, concat_dim=aggdims[-1] \
                                            , data_vars='minimal') )
    return xr.concat(ncs,dim=aggdims[0],data_vars='minimal')    

