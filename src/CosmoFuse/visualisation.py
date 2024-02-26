import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def contours(ax, xplot, yplot, labels, truths, bins=200, smooth=0.5):
    
    hist, xedges, yedges = np.histogram2d(xplot, yplot, bins)
    hist = hist.T

    Histsmooth = gaussian_filter(hist, smooth)

    Ngrid = len(xedges) - 1

    xbin = np.zeros(Ngrid, dtype=float)
    ybin = np.zeros(Ngrid, dtype=float)

    i = 0
    while i < Ngrid:
        xbin[i] = (xedges[i + 1] + xedges[i]) / 2.0
        ybin[i] = (yedges[i + 1] + yedges[i]) / 2.0
        i += 1

    mult = np.sort(Histsmooth, axis=None)

    p = ([0.99, 0.95, 0.68])
    clevels = p

    Nlevels = len(p)
    m = 0
    while (m < Nlevels):
        i = len(mult) - 1
        sum = 0.0
        while (sum <= p[m] * xplot.size):
            sum = sum + mult[i]
            i = i - 1
        imin = i + 1
        clevels[m] = mult[imin]
        m += 1

    X, Y = np.meshgrid(xbin, ybin)
 

    
    #ax.title("Posterior Contours", fontsize=22)
    ax.set_xlabel(labels[0], fontsize=20)
    ax.set_ylabel(labels[1], fontsize=20)
    ax.contourf(X, Y, Histsmooth, 40, cmap='Spectral_r')
    CS = ax.contour(X, Y, Histsmooth, np.sort(clevels), colors='k')


    fmt = {}
    strs = [r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"]
    for l, s in zip(CS.levels, strs):
        fmt[l] = s

    ax.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=22)
    ax.scatter(truths[0], truths[1], marker='x', color='k', s=100, label="True Value")
    
def make_corner_plot(dist, param_names, theta_obs, smooth=0.5):
    ndims = dist.shape[1]
    fig,ax = plt.subplots(ndims,ndims, figsize=(12,12))

    for i in range(ndims):
        for j in range(i+1):
            if i==j:
                ax[i,j].hist(dist[:,i], bins='auto', density=True)
                ax[i,j].axvline(theta_obs[i], c='k', ls='--')
                ax[i,j].set_xlabel(param_names[i], fontsize=20)
                continue
            
            contours(ax[i,j], dist[:,j], dist[:,i], [param_names[j], param_names[i]], [theta_obs[j], theta_obs[i]], smooth=smooth)
            fig.delaxes(ax[j,i])
            
    fig.tight_layout()

def make_corner_plot(dist, param_names, theta_obs, smooth=0.5):
    ndims = dist.shape[1]
    fig,ax = plt.subplots(ndims,ndims, figsize=(ndims*6,ndims*6))

    for i in range(ndims):
        for j in range(i+1):
            if i==j:
                ax[i,j].hist(dist[:,i], bins='auto', density=True)
                ax[i,j].axvline(theta_obs[i], c='k', ls='--')
                ax[i,j].set_xlabel(param_names[i], fontsize=20)
                continue
            
            contours(ax[i,j], dist[:,j], dist[:,i], [param_names[j], param_names[i]], [theta_obs[j], theta_obs[i]], smooth=smooth)
            fig.delaxes(ax[j,i])
            
    fig.tight_layout()
    
    return fig, ax
    
    