''' return the pressure integrated from the surface
takes into account surface elevation
uses cumulative trapezoidale integration
first value below the surface corresponds to quadratic integration 
implicit value at the surface is zero (must add atmospheric value to obtain true value)
NJAL May 2017
'''
from scipy.integrate import cumtrapz

grav = 9.81

def get_pres(rho,zr,zeta=0,axis=-1):
    if rho.ndim == 1:
        return -grav*(cumtrapz(rho[::-1],zr[::-1],initial=0)[::-1]\
                    + rho[-1]*(zr[-1]-zeta) )
    else:
        if axis != -1:
            rho = rho.swapaxes(axis,-1)
            zr = zr.swapaxes(axis,-1)
        return -grav*(cumtrapz(rho[...,::-1],zr[...,::-1],initial=0)[...,::-1] \
                + (rho[...,-1]*(zr[...,-1]-zeta))[...,None]).swapaxes(axis,-1)
        
def get_redpres(buoy,zr,zeta=0,axis=-1):
    ''' Reduced pressure (do not multiply by grav, take buoyancy as input) '''
    if buoy.ndim == 1:
        return cumtrapz(buoy[::-1],zr[::-1],initial=0)[::-1]\
                    + buoy[-1]*(zr[-1]-zeta)
    else:
        if axis != -1:
            buoy = buoy.swapaxes(axis,-1)
            zr = zr.swapaxes(axis,-1)
        return ( cumtrapz(buoy[...,::-1],zr[...,::-1],initial=0)[...,::-1] \
                + (buoy[...,-1]*(zr[...,-1]-zeta))[...,None]).swapaxes(axis,-1)
        
