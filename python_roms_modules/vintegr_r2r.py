import numpy as np

def vintsurf_r2r(field,zr,zw):
    ''' Vertical integration from the surface of a field defined on rho points 
    vertical axis for the array is supposed to be the last one (shape does not matter)
    rho-points are assumed to be the middle of w-points -- this is a good approximation 
    only at depth''' # nop, correct that when you get a fixed version
    lashape = field.shape
    Hzt = zr.T - zw.T[1:,:]
    Hzb = zw.T[:-1,:] - zr.T
    res = np.zeros(field.T.shape)
    if field.ndim ==1:
        res[-1] = field.T[-1]*Hzt[-1]
        for iz in range(lashape[-1]-1,0,-1):
            res[iz-1] = res[iz] + field.T[iz]*Hzb[iz] + field.T[iz-1]*Hzt[iz-1]
    else:
        res[-1,:] = field.T[-1,:]*Hzt[-1,:]
        for iz in range(lashape[-1]-1,0,-1):
            res[iz-1,:] = res[iz,:] + field.T[iz,:]*Hzb[iz,:] + field.T[iz-1,:]*Hzt[iz-1,:]
    return res.T
