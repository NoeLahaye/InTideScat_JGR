''' hor_subsamp_ncfile.py
read variables in a given netCDF file, store horizontal subsampled version
WARNING: entire variables will be loaded in memory, be sure its not too big to avoid swapping
Possibility to add topography (or any 2D field) on corresponding subsampled grid (as well as topo meta info) to be able to reconstruct z-grid on this subgrid
NJAL February 2018 '''
from __future__ import print_function, division
import numpy as np
import os
from netCDF4 import Dataset
import def_chunk

simul = "luckyt"
var = "buoy"

pathbase = "/".join([os.getenv("SCRATCHDIR"),simul.upper(),"data/tseries/"])
filepath = pathbase+"{0}_lowf_{1}.nc".format(simul,var)
filestor = filepath.replace("lowf","subsamp").replace(var,"stratif")
doflipt = True      # flip time dimension, switch from fast to slow dim
addtopo = True      # add topography and vertical grid infos
doappend = True     # file will be overwritten if False 
nsx = 3             # subsampling step

if addtopo:
    gfilename = os.getenv('STOREDIR')+"/lucky_corgrd.nc"
    zfilename = pathbase+"{}_tseries_2Dvars.nc".format(simul)

ncr = Dataset(filepath,'r')

Nx, Ny = ncr.dimensions['xi_rho'].size, ncr.dimensions['eta_rho'].size
(ixs, iys), (nx, ny) = def_chunk.get_indchk((Nx,Ny),(1,1),nsx)
nx, ny = nx[0], ny[0]

if not (os.path.exists(filestor) and doappend):
    ncw = Dataset(filestor,'w')
else:
    ncw = Dataset(filestor,'a')
# copy non-existing dimensions and associated coordinate variables
for dim in ncr.dimensions.values():
    if dim.name not in ncw.dimensions.keys():
        print("creating dim and copying coord. variable",dim.name)
        if dim.name == "xi_rho":
            ncw.createDimension(dim.name,nx)
            xi = ncr.variables[dim.name][ixs[0]::nsx]
            ncw.createVariable(dim.name,"i",(dim.name,))[:] = xi
        elif dim.name == "eta_rho":
            ncw.createDimension(dim.name,ny)
            eta = ncr.variables[dim.name][iys[0]::nsx]
            ncw.createVariable(dim.name,"i",(dim.name,))[:] = eta                  
        else:
            ncw.createDimension(dim.name,dim.size)
            ncw.createVariable(dim.name,"i",(dim.name,))[:] = ncr.variables[dim.name][:]
# copy variables (discard coordinate variables)
for vnam,vvar in ncr.variables.items():
    if vnam not in ncr.dimensions.keys() and vnam not in ncw.variables.keys():
        if "xi_rho" not in vvar.dimensions:
            ncw.createVariable(vnam,'f',vvar.dimensions)[:] = vvar[:]
        else:
            if vvar.ndim == 2:
                ncw.createVariable(vnam,'f',vvar.dimensions)[:] = \
                    vvar[iys[0]::nsx,ixs[0]::nsx]
            elif vvar.ndim == 3:
                ncw.createVariable(vnam,'f',vvar.dimensions)[:] = \
                    vvar[:,iys[0]::nsx,ixs[0]::nsx]
            elif vvar.ndim == 4:
                if doflipt:
                    ncw.createVariable(vnam,'f',vvar.dimensions[-1:]+vvar.dimensions[:-1])[:]\
                        = np.rollaxis(vvar[:][:,iys[0]::nsx,ixs[0]::nsx,:],3)
                else:
                    ncw.createVariable(vnam,'f',vvar.dimensions)[:] = \
                                            vvar[:][:,iys[0]::nsx,ixs[0]::nsx,:]
            else:
                print("case not handled for variable",vnam)
                continue
        print("copied",vnam)

ncr.close()

if addtopo and (not doappend or 'h' not in ncw.variables.keys()):
    ncr = Dataset(gfilename,'r')
    lavar = ncr.variables['h']
    xi = ncw.variables['xi_rho'][:]
    eta = ncw.variables['eta_rho'][:]
    ncw.createVariable(lavar.name,'f',lavar.dimensions)[:] = lavar[eta,xi]
    ncr.close()
    print("copied topo")
    ncr = Dataset(zfilename,'r')
    ncw.hc = ncr.hc
    ncw.Cs_r = ncr.Cs_r
    ncw.Cs_w = ncr.Cs_w
    ncr.close()

ncw.close()

