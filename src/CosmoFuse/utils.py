import numpy as np
import healpy as hp
import matplotlib.pylab as pylab

def pixel2RaDec(pixel_indices, NSIDE):
    theta, phi = hp.pixelfunc.pix2ang(NSIDE, pixel_indices, nest=False)
    ra = phi
    dec = np.pi/2.0-theta
    return ra, dec

def set_mpl_params():
    
    params = {
        'figure.figsize': (12, 8),
        'legend.fontsize': '20',
        'axes.labelsize': '20',
        'axes.titlesize':'24',
        'xtick.labelsize':'18',
        'ytick.labelsize':'18'}
    
    pylab.rcParams.update(params)
    
def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])  

