''' routines that compute vertical level (z_levels and rho_levels)
adapt to the shape of input hbot, zeta -- zdimension is appended (last axis)
NJAL June 2017 '''
from __future__ import division
import numpy as np

def zlev_rho(hbot,zeta,hc,Cs,slevs=None):
    """ Compute depth of sigma-levels at rho points (vertically)
    either all levels (size of Cs is used) or only those given in slevs, if provided 
    """
    nz = len(Cs)
    if slevs is None: 
        slevs = np.arange(nz)
    elif isinstance(slevs,int):
        slevs = np.array([slevs])
    elif isinstance(slevs,list):
        slevs = np.array(slevs)
    cff = hc/nz*(slevs+1-nz-0.5)
    if not isinstance(hbot, np.ndarray): hbot = np.array(hbot)
    if not isinstance(zeta, np.ndarray): zeta = np.array(zeta)
    return zeta[...,None] + (zeta[...,None]+hbot[...,None])\
            *(cff + Cs[slevs]*hbot[...,None])/(hbot[...,None]+hc)

def zlev_w(hbot,zeta,hc,Cs,slevs=None):
    """ Compute depth of sigma-levels at w points (vertically)
    either all levels (size of Cs is used) or only those given in slevs, if provided 
    """
    nz = len(Cs)-1
    if slevs is None: 
        slevs = np.arange(nz+1)
    elif isinstance(slevs,int):
        slevs = np.array([slevs])
    elif isinstance(slevs,list):
        slevs = np.array(slevs)
    if not isinstance(hbot, np.ndarray): hbot = np.array(hbot)
    if not isinstance(zeta, np.ndarray): zeta = np.array(zeta)
    zw = np.empty(hbot.shape+(len(slevs),))
    if 0 in slevs:
        cff = hc/nz*(slevs[1:]-nz)
        zw[...,0] = -hbot
        zw[...,1:] = zeta[...,None] + (zeta[...,None]+hbot[...,None])\
            *(cff + Cs[slevs[1:]]*hbot[...,None])/(hbot[...,None]+hc)
    else:
        cff = hc/nz*(slevs-nz)
        zw = zeta[...,None] + (zeta[...,None]+hbot[...,None])\
            *(cff + Cs[slevs]*hbot[...,None])/(hbot[...,None]+hc)
    return zw

#def zlev_wold(hbot,zeta,hc,Cs,slevs=None):
    #""" Compute depth of sigma-levels at w points (vertically)
    #either all levels (size of Cs is used) or only those given in slevs, if provided 
    #this version has error if slevs has 0
    #"""
    #nz = len(Cs)-1
    #if slevs is None: 
        #slevs = np.arange(nz+1)
    #elif isinstance(slevs,int):
        #slevs = np.array([slevs])
    #elif isinstance(slevs,list):
        #slevs = np.array(slevs)
    #zw = np.empty(hbot.shape+(len(slevs),))
    #if 0 in slevs:
        #cff = hc/nz*(slevs[:-1]+1-nz)
        #zw[...,0] = -hbot
        #zw[...,1:] = zeta[...,None] + (zeta[...,None]+hbot[...,None])\
            #*(cff + Cs[slevs[1:]]*hbot[...,None])/(hbot[...,None]+hc)
    #else:
        #cff = hc/nz*(slevs-nz)
        #zw = zeta[...,None] + (zeta[...,None]+hbot[...,None])\
            #*(cff + Cs[slevs]*hbot[...,None])/(hbot[...,None]+hc)
    #return zw

def dz_anyND(hbot,zeta,hc,dCs,N):
    """ return vertical level thickness from Delta Cs array
    shape is shape of hbot,zeta * shape of dCs
    """
    hshape = hbot.shape
    zshape = dCs.size,
    if not isinstance(hbot, np.ndarray): hbot = np.array(hbot)
    if not isinstance(zeta, np.ndarray): zeta = np.array(zeta)
    dz = ((zeta + hbot)/(hc + hbot)).ravel()[:,None]*(hc/N+dCs[None,:]*hbot.ravel()[:,None])
    return dz.reshape(hshape + zshape)

def dz_any1D(hbot,zeta,hc,dCs,N):
    """ same as dz_anyND but assume entries are flat arrays of same shape (fancy addressing, 
    staggered access) or scalar. Just compute the product
    """
    return (zeta + hbot)*(hc/N+dCs*hbot)/(hc + hbot)
    
def get_scoord(hc,theta_s,theta_b,Nz):
    ''' get_scoord(hc,theta_s,theta_b,Nz)
    copied from set_scoord.F, NEW_S_COORD activated. Returns sigma levels at rho and w points
    '''
    ds = 1./Nz
    scr = ds*(np.arange(1,Nz+1)-Nz-0.5)
    scw = ds*(np.arange(Nz+1)-Nz)
    if theta_s > 0:
        csr = (1.-np.cosh(theta_s*scr))/(np.cosh(theta_s)-1.)
        csw = (1.-np.cosh(theta_s*scw))/(np.cosh(theta_s)-1.)
    else:
        csr, csw = -scr**2, -scw**2
    if theta_b > 0:
        Cs_r = (np.exp(theta_b*csr)-1.) / (1.-np.exp(-theta_b))
        Cs_w = (np.exp(theta_b*csw)-1.) / (1.-np.exp(-theta_b))
    else:
        Cs_r, Cs_w = csr, csw

    return Cs_r, Cs_w

