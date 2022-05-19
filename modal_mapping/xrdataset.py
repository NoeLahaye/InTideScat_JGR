import xarray as xr
#import time

def xrdataset(filenames,aggdims,no_mf=True):
    """ filenames: shape 2 list [dim y [dim x]] ; aggdims = [dimy,dimx] """
    ncs = []
    for names in filenames:
        #tmes, tmeb = time.clock(), time.time()
        if no_mf:
            ncc = [xr.open_dataset(name) for name in names]
            ncs.append( \
                    xr.concat(ncc, dim=aggdims[-1], data_vars='minimal') )
        else:
            ncs.append( xr.open_mfdataset(names, concat_dim=aggdims[-1] \
                                            , data_vars='minimal') )
        #print("done with files {}...".format(names[0]),time.clock()-tmes,time.time()-tmeb)
    return xr.concat(ncs,dim=aggdims[0],data_vars='minimal')    

