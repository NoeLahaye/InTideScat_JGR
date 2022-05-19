from __future__ import division
import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt as filt

def butterfilt(tt,data,axis=-1,cutoff=1,mode='low'):
	''' Order 4 butterworth forward-backward filtering with gustafsson's method for endpoints
	evenly spaced x-points are assumed (dt=cst)
	cutof is roughly the inertial frequency if not specified, and mode is low-pass'''
	if isinstance(tt,(float,int,np.float32,np.float64)):
		dt = tt
	else:
		dt = tt[1]-tt[0]
	Nyquist = 1/2./dt
	b, a = butter(4,cutoff/Nyquist,mode)
	#return filt(b,a,data,axis=axis,method="gust")
	return filt(b,a,data,axis=axis) # curie (old numpy)

