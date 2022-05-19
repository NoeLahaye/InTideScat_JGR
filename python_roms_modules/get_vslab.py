from __future__ import print_function, division
import numpy as np
import R_smooth as sm
from matplotlib import pyplot as plt
 
Eradius = 6371315.0     # Earth radius as used in Croco (cf. scalars.h), m

def get_vslab(xm,ym,Lslab,lcenter,dl,ang,xx,yy,doverb=True):
    
    # define lon, lat points ov vertical section
    (indxm,indym) = np.unravel_index(((xx-xm)**2+(yy-ym)**2).argmin(),xx.shape)
    Nl = int(Lslab/dl) + int((Lslab//dl)%2==0)   # odd
    lonslab = xm + np.arange(-(Nl//2)*dl-lcenter,(Nl//2+1)*dl-lcenter,dl)\
                            /Eradius*np.cos(ang)*180./np.pi
    latslab = ym + np.arange(-(Nl//2)*dl-lcenter,(Nl//2+1)*dl-lcenter,dl)\
                            /Eradius*np.sin(ang)*180./np.pi
    xslab = (np.arange(-(Nl//2),Nl//2+1)*dl-lcenter)*np.cos(ang)
    yslab = (np.arange(-(Nl//2),Nl//2+1)*dl-lcenter)*np.sin(ang)
    lslab = np.arange(-(Nl//2),Nl//2+1)*dl-lcenter
    if doverb:
        print('min & max lon.:',lonslab.min(),lonslab.max(),'shape:',lonslab.shape)
        print('min & max lat.:',latslab.min(),latslab.max(),'shape:',latslab.shape)
    
    # find points in parent grid involved for interp on child grid
    cmax = list(xx.shape)
    elem,coef = sm.get_tri_coef(xx,yy,lonslab.reshape(-1,1),\
                                    latslab.reshape(-1,1))
    print(elem.shape,coef.shape,Nl)
    indx,indy = np.unravel_index(elem.reshape(Nl,3),cmax)
    coords = [max([0,indy.min()-1]),min([indy.max()+2,cmax[1]]),\
                max([0,indx.min()-1]),min([indx.max()+2,cmax[0]])]; # add one edge
    subelem = np.ravel_multi_index((indx-coords[2],indy-coords[0]),\
                (coords[3]-coords[2],coords[1]-coords[0]))
    
    return (indxm,indym),(lonslab,latslab),(xslab,yslab),lslab,subelem,coef[:,0,:],coords
    #return (indxm,indym),(lonslab,latslab),(xslab,yslab),lslab,subelem,coef,coords

def show_vslab(xms,yms,Lslabs,lcenters,angs,xx,yy,topo=None):
    plt.ion()
    fig,ax = plt.subplots()
    if topo is not None:
        plt.pcolormesh(xx,yy,-topo,cmap='terrain')
        plt.colorbar()
    if isinstance(xms,(int,float)): xms = [xms]
    if isinstance(yms,(int,float)): yms = [yms]
    if isinstance(Lslabs,(int,float)): Lslabs = [Lslabs]
    if isinstance(lcenters,(int,float)): lcenters = [lcenters]
    if isinstance(angs,(int,float)): angs = [angs]
        
    for xm,ym,Lslab,lcenter,ang in zip(xms,yms,Lslabs,lcenters,angs):
        (indxm,indym) = np.unravel_index(((xx-xm)**2+(yy-ym)**2).argmin(),xx.shape)
        lonslab = xm + (np.array([-1,1])*Lslab/2-lcenter)/Eradius*np.cos(ang)\
                            *180./np.pi
        latslab = ym + (np.array([-1,1])*Lslab/2-lcenter)/Eradius*np.sin(ang)\
                            *180./np.pi
        plt.plot(lonslab,latslab,color='k',linewidth=2)
    plt.title('North MAR: bathy. and sections')
    plt.axis([xx[0,:].max(),xx[-1,:].min(),yy[:,0].max(),yy[:,-1].min()])
    plt.grid(True)

    return fig,ax
         
def def_vslab(xm,ym,Lslab,ang,lcenter=0,doverb=False):
    
    # define lon, lat points ov vertical section
    lonslab = xm + (np.array([-1,1])*Lslab/2-lcenter)\
                            /Eradius*np.cos(ang)*180./np.pi
    latslab = ym + (np.array([-1,1])*Lslab/2-lcenter)\
                            /Eradius*np.sin(ang)*180./np.pi
    return lonslab, latslab 

