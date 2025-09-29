import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde


def contours(
    ax,
    xplot,
    yplot,
    labels,
    truths=None,
    fill=True,
    colour="k",
    bins=200,
    smooth=0.5,
    lw=1,
):
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

    p = [0.99, 0.95, 0.68]
    clevels = p

    Nlevels = len(p)
    m = 0
    while m < Nlevels:
        i = len(mult) - 1
        sum = 0.0
        while sum <= p[m] * xplot.size:
            sum = sum + mult[i]
            i = i - 1
        imin = i + 1
        clevels[m] = mult[imin]
        m += 1

    X, Y = np.meshgrid(xbin, ybin)

    if fill:
        ax.contourf(X, Y, Histsmooth, 40, cmap="Spectral_r")
    CS = ax.contour(X, Y, Histsmooth, np.sort(clevels), colors=colour, linewidths=lw)

    fmt = {}
    strs = [r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"]
    for l, s in zip(CS.levels, strs):
        fmt[l] = s

    ax.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=22)
    if truths is not None:
        ax.scatter(truths[0], truths[1], marker="x", color="k", s=100)


def make_corner_plot(
    dist,
    param_names,
    theta_obs=None,
    fill=True,
    colour="k",
    fig_ax=None,
    smooth=0.5,
    label=None,
    kde=False,
    result=False,
    fontsize=16,
    nbins=200,
):
    ndims = dist.shape[1]
    if fig_ax is None:
        fig, ax = plt.subplots(ndims, ndims, figsize=(ndims * 6, ndims * 6))
        fig.subplots_adjust(wspace=0.0, hspace=0.0)

    else:
        fig, ax = fig_ax

    for i in range(ndims):
        for j in range(i + 1):
            if j == 0:
                if i != 0:
                    ax[i, j].set_ylabel(param_names[i], fontsize=fontsize + 10)
            else:
                ax[i, j].set_yticklabels([])
            if i == ndims - 1:
                ax[i, j].set_xlabel(param_names[j], fontsize=fontsize + 10)
            else:
                ax[i, j].set_xticklabels([])

            ax[i, j].tick_params(
                axis="both",
                which="major",
                size=12,
                direction="inout",
                labelsize=fontsize - 4,
            )
            ax[i, j].tick_params(axis="both", which="minor", size=10, direction="inout")

            if i == j:
                if fill:
                    if i == 0:
                        counts, binedges, _ = ax[i, j].hist(
                            dist[:, i],
                            bins="auto",
                            density=True,
                            color=colour,
                            label=label,
                        )
                    else:
                        counts, binedges, _ = ax[i, j].hist(
                            dist[:, i], bins="auto", density=True, color=colour
                        )
                else:
                    if i == 0:
                        counts, binedges, _ = ax[i, j].hist(
                            dist[:, i],
                            bins="auto",
                            histtype="step",
                            density=True,
                            color=colour,
                            label=label,
                            lw=fontsize / 15,
                        )
                    else:
                        counts, binedges, _ = ax[i, j].hist(
                            dist[:, i],
                            bins="auto",
                            histtype="step",
                            density=True,
                            color=colour,
                            lw=fontsize / 15,
                        )

                if theta_obs is not None:
                    ax[i, j].axvline(theta_obs[i], c="k", ls="--")

                if result:
                    bins = (binedges[1:] + binedges[:-1]) / 2
                    best = bins[np.argmax(counts)]
                    var = np.std(dist[:, i])
                    ax[i, j].set_title(
                        rf"{param_names[i]}$ = {best:.2f} \pm {var:.2f}$",
                        fontsize=fontsize + 2,
                    )

                continue

            if theta_obs is not None:
                contours(
                    ax[i, j],
                    dist[:, j],
                    dist[:, i],
                    [param_names[j], param_names[i]],
                    [theta_obs[j], theta_obs[i]],
                    colour=colour,
                    fill=fill,
                    smooth=smooth,
                    lw=fontsize / 15,
                    bins=nbins,
                )
            else:
                contours(
                    ax[i, j],
                    dist[:, j],
                    dist[:, i],
                    [param_names[j], param_names[i]],
                    colour=colour,
                    fill=fill,
                    smooth=smooth,
                    lw=fontsize / 15,
                    bins=nbins,
                )
            if fig_ax is None:
                fig.delaxes(ax[j, i])

    return fig, ax
