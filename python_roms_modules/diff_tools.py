from __future__ import division
import numpy as np

def diff_2d_ongrid(var,pm,axis=0):
    nx, ny = var.shape
    slix1 = slice(0,nx-int(axis==0)); slix2 = slice(int(axis==0),nx)
    sliy1 = slice(0,ny-int(axis==1)); sliy2 = slice(int(axis==1),ny)
    if isinstance(pm,(int,float,np.float32,np.float64)):
        dvar = (diff_2d_offgrid(var[slix1,sliy1],pm,axis=axis) \
            + diff_2d_offgrid(var[slix2,sliy2],pm,axis=axis))/2.
    elif pm.ndim==1:
        dvar = (diff_2d_offgrid(var[slix1,sliy1],pm[:-1],axis=axis) \
            + diff_2d_offgrid(var[slix2,sliy2],pm[1:],axis=axis))/2.
    else:
        dvar = (diff_2d_offgrid(var[slix1,sliy1],pm[slix1,sliy1],axis=axis) \
            + diff_2d_offgrid(var[slix2,sliy2],pm[slix2,sliy2],axis=axis))/2.
    return dvar

def diff_2d_offgrid(var,pm,axis=0):
    nx, ny = var.shape
    slix1 = slice(0,nx-int(axis==0)); slix2 = slice(int(axis==0),nx)
    sliy1 = slice(0,ny-int(axis==1)); sliy2 = slice(int(axis==1),ny)
    if isinstance(pm,(int,float,np.float32,np.float64)):
        dvar = (var[slix2,sliy2] - var[slix1,sliy1])*pm
    elif pm.ndim==1:
        dvar = (var[slix2,sliy2] - var[slix1,sliy1]) * (pm[1:] + pm[:-1])/2.
    else:
        dvar = (var[slix2,sliy2] - var[slix1,sliy1]) \
                * (pm[slix1,sliy1]+pm[slix2,sliy2])/2.
    return dvar

        
#def diff_o2(tab,pm,axis=0):
    #ndim = tab.ndim
    #nns = tab.shape
    #slic1 = []; slic2 = []
    #for ii,nn in zip(ndim,nns):
        #if ii==axis:
            #slic1.append(0,nn-2)
            #slic2.append(2,nn)
        #else:
            #slic1.append(0,nn)
            #slic2.append(0,nn)
    #slic1 = tuple(slic1)
    #slic2 = tuple(slic2)
    #if pm.shape == nns:
        #return 0.5*(tab[slic2]-tab[slic1])
