from __future__ import print_function
import numpy as np

def submat_from_pos(xpos,ypos,Nx,Ny,xgrid,ygrid,**kwargs):
    # find indices of subdomain with size Nx,Ny centered around xpos,ypos 
    # in 2D xgrid and ygrid (not regular, e.g. curvilinear coordinates)
    if 'verbose' in kwargs: verbose = kwargs['verbose'] 
    else: verbose = True

    (indx0,indy0) = np.unravel_index(((xgrid-xpos)**2+(ygrid-ypos)**2).argmin(),\
                                            xgrid.shape)
    hcoord = [indy0-Ny//2,indy0+Ny//2,indx0-Nx//2,indx0+Nx//2]
    if hcoord[0] < 0:
        if verbose: print('lower y bound out of domain -- correcting')
        hcoord[0] = 0
        Ny = hcoord[1]
    if hcoord[2] < 0:
        if verbose: print('lower x bound out of domain -- correcting')
        hcoord[2] = 0
        Nx = hcoord[3]
    if hcoord[1] > xgrid.shape[1]:
        if verbose: print('upper y bound out of domain -- correcting')
        hcoord[1] = xgrid.shape[1]
        Ny = hcoord[1]-hcoord[0]
    if hcoord[3] > xgrid.shape[0]:
        if verbose: print('upper x bound out of domain -- correcting')
        hcoord[3] = xgrid.shape[0]
        Nx = hcoord[3]-hcoord[2] 
    if verbose: print('coordinates of horizontal domain:',hcoord)
    return indx0,indy0,hcoord 
