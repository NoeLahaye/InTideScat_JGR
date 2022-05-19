from __future__ import print_function, division
import numpy as np
from R_tools import rho2u,rho2v

def calc_etat(ubar,vbar,hflow,pm,pn):
    ''' compute divergence of barotropic momentum (units m/h) 
    arrays are (x,y) ordered -- hflow is full column depth (SSE-z_bot)'''
    return -( np.diff(rho2u(hflow/pn)*ubar,axis=0)[:,1:-1] \
        + np.diff(rho2v(hflow/pm)*vbar,axis=1)[1:-1,:] )\
        * pm[1:-1,1:-1]*pn[1:-1,1:-1]*3600.

