"""
TTV and RV Analysis Module

This module provides utility functions for modeling and visualizing
Transit Timing Variations (TTVs) and Radial Velocity (RV) signals
in exoplanetary systems. Includes functionality for data preprocessing,
parameter handling, MCMC analysis, and plotting.

Dependencies:
    astropy, numpy, matplotlib, ttvfast
"""

from astropy import constants as const
from numpy import log, argmax, array, zeros, sin, cos, pi, exp
from ttvfast import models
import matplotlib.pyplot as plt

# Physical constants
mearth = const.M_earth.cgs.value  # grams
msun = const.M_sun.cgs.value      # grams
au_per_day = 1731460              # meters per second
mstar = 1.034                     # solar masses


def sun_to_earth(solar_mass):
    """
    Convert solar mass to Earth mass.

    Args:
        solar_mass (float): Mass in units of solar masses.

    Returns:
        float: Mass in Earth masses.
    """
    return solar_mass / msun * mearth


def best_fit(x, y):
    """
    Compute the best-fit linear model y = mx + b.

    Args:
        x (list or array): Independent variable values.
        y (list or array): Dependent variable values.

    Returns:
        tuple: (slope, intercept) of the best-fit line.
    """
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    n = len(x)
    numer = sum(xi * yi for xi, yi in zip(x, y)) - n * xbar * ybar
    denom = sum(xi**2 for xi in x) - n * xbar**2
    m = numer / denom
    b = ybar - m * xbar
    return m, b


def ttvs(measured, epoch):
    """
    Compute Transit Timing Variations (TTVs) as observed minus calculated times.

    Args:
        measured (list of lists): Observed transit midpoints.
        epoch (list of lists): Corresponding epochs.

    Returns:
        list: O-C (observed - calculated) timing variations.
    """
    oc = []
    for i in range(len(measured)):
        slope, intercept = best_fit(epoch[i], measured[i])
        y_fit = intercept + slope * array(epoch[i])
        oc.append(measured[i] - y_fit)
    return oc


def flatten_list(_2d_list):
    """
    Flatten a 2D list to a 1D list.

    Args:
        _2d_list (list of lists): Nested list.

    Returns:
        list: Flattened list.
    """
    flat = []
    for element in _2d_list:
        flat.extend(element if isinstance(element, list) else [element])
    return flat


def trim(nplanets, epoch, measured, model, error, flatten=True):
    """
    Trim time series data to ensure consistent lengths across planets.

    Args:
        nplanets (int): Number of planets.
        epoch (list): Epochs of measured transits.
        measured (list): Measured transit times.
        model (list): Modeled transit times.
        error (list): Errors on measurements.
        flatten (bool): If True, return flattened arrays.

    Returns:
        tuple: Trimmed model, measured, error, and epoch values.
    """
    mod, meas, err, ep = [], [], [], []

    for i in range(nplanets):
        len_model = len(model[i])
        len_meas = len(measured[i])
        if len_meas > len_model:
            meas.append(measured[i][:len_model])
            mod.append(model[i])
            err.append(error[i][:len_model])
            ep.append(epoch[i][:len_model])
        else:
            meas.append(measured[i])
            mod.append(model[i][:len_meas])
            err.append(error[i][:len_meas])
            ep.append(epoch[i][:len_meas])

    if flatten:
        return (array(flatten_list(mod)),
                array(flatten_list(meas)),
                array(flatten_list(err)),
                array(flatten_list(ep)))
    return mod, meas, err, ep


def generate_planets(theta, system):
    """
    Generate Planet objects from system and theta parameters.

    Args:
        theta (list): Sampled orbital parameters.
        system (object): System object with fixed and variable parameters.

    Returns:
        list: List of Planet objects.
    """
    nplanets = system.nplanets_rvs
    planet_labels = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    keys = ['mass', 'period', 'eccentricity', 'inclination',
            'longnode', 'argument', 'mean_anomaly']

    elements = dict(zip(system.fixed_labels, system.fixed))
    elements.update(zip(system.variable_labels, theta))

    planets = []
    for j in range(nplanets):
        label = planet_labels[j]
        params = {key: elements[f"{key}_{label}"] for key in keys}
        planet = models.Planet(**params)
        planets.append(planet)

    return planets


def plot_periodogram(t, rv, title):
    """
    Plot a Lomb-Scargle-like periodogram for RV data.

    Args:
        t (list): Observation times.
        rv (list): Radial velocities.
        title (str): Title of the plot.
    """
    t = array(t) - t[0]
    rv = array(rv)
    s1 = len(t)
    sx = sum(rv) / s1
    sx2 = sum((rv - sx) ** 2) / s1

    pstart, pstop, pint = 60, t[-1], 1
    ni = -6.362 + 1.193 * s1 + 0.00098 * s1 ** 2
    pw = zeros(((pstop - pstart) // pint, 3))

    for m in range(pw.shape[0]):
        pt = pstart + (m + 1) * pint
        w = 2 * pi / pt
        tao = 0.0
        p1 = sum(rv * cos(w * (t - tao)))
        p3 = sum(rv * sin(w * (t - tao)))
        p2 = sum(cos(w * (t - tao))**2)
        p4 = sum(sin(w * (t - tao))**2)
        px = 0.5 * (p1 ** 2 / p2 + p3 ** 2 / p4) / sx2
        pw[m] = [pt, px, 1 - (1 - exp(-px)) ** ni]

    fig, ax1 = plt.subplots()
    ax1.plot(pw[:, 0], pw[:, 1], 'b', alpha=0.9, label='Power')
    ax1.set_xlabel('Period (day)', fontsize=16)
    ax1.set_ylabel('Power', fontsize=16)
    ax1.set_xlim([60, 700])
    ax1.set_xticks([90, 180, 360, 540])

    ax2 = ax1.twinx()
    ax2.plot(pw[:, 0], pw[:, 2], 'y', alpha=0.9, label='False Alarm Probability')
    ax2.set_ylabel('False Alarm Probability', fontsize=16)

    lines, labels = ax1.get_legend_handles_labels() + ax2.get_legend_handles_labels()
    ax1.legend(lines, labels, bbox_to_anchor=(0.78, 0.9))
    ax1.set_title(title)
    plt.close('all')


def sampler_to_theta_max(run):
    """
    Extract the sample with the highest log-probability.

    Args:
        run (object): MCMC run object with sampler and thin attribute.

    Returns:
        list: Theta corresponding to the max log-prob.
    """
    samples = run.sampler.get_chain(flat=True, thin=run.thin)
    run.theta_max = samples[argmax(run.sampler.get_log_prob(flat=True, thin=run.thin))]
    return run.theta_max


def bic(run):
    """
    Compute Bayesian Information Criterion (BIC) for the MCMC run.

    Args:
        run (object): MCMC run object with sampler, system, and theta_max.

    Returns:
        float: BIC value.
    """
    k = len(run.theta_max)
    n = sum(len(t) for t in run.system.measured) + len(run.system.rvbjd)
    lnprob = argmax(run.sampler.get_log_prob(flat=True, thin=run.thin))
    run.bic = k * log(n) - 2 * lnprob
    return run.bic


def plot_ttvs(nplanets, measured, epoch, error, model, model_epoch, filename=None, silent=False):
    """
    Plot Transit Timing Variations (TTVs) for multiple planets.

    Args:
        nplanets (int): Number of planets.
        measured (list): Measured mid-transit times.
        epoch (list): Epochs of measured transits.
        error (list): Errors of measured midpoints.
        model (list): Modeled mid-transit times.
        model_epoch (list): Epochs of modeled transits.
        filename (str, optional): File to save the figure.
        silent (bool): If False, display the plot interactively.
    """
    fig, axes = plt.subplots(nplanets, figsize=(20, 16), sharex=True)
    fig.suptitle('TTVs', fontsize=30)
    planet_labels = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    ttv_model = ttvs(model, model_epoch)
    ttv_data = ttvs(measured, epoch)

    for i in range(nplanets):
        ax = axes[i]
        ax.set_title(f'O-C Kepler 36-{planet_labels[i]} with Fitted Orbital Elements', fontsize=20)
        ax.errorbar(epoch[i], ttv_data[i], error[i], fmt='o', label='TTV From Measured Transits')
        ax.plot(model_epoch[i], ttv_model[i], label='Best Fit (O-C)')
        ax.legend(fontsize=16)

    if filename:
        fig.savefig(filename)
    if not silent:
        plt.show()
    plt.close('all')


def plot_rvs(bjd, rv, rv_err, rv_model, filename, silent=False):
    """
    Plot Radial Velocity (RV) data and residuals.

    Args:
        bjd (list): Barycentric Julian Dates of measurements.
        rv (list): Observed radial velocities.
        rv_err (list): Measurement uncertainties.
        rv_model (list): Modeled radial velocities.
        filename (str): File path to save the plot.
        silent (bool): If False, display the plot.
    """
    rv_resid = rv - rv_model
    fig = plt.figure(1)

    # RV data
    frame1 = fig.add_axes((.1, .3, .8, .6))
    frame1.set_title('RVs')
    frame1.set_ylabel('RV [m/s]')
    frame1.set_xlabel('BJD')
    plt.errorbar(bjd, rv, rv_err, fmt='o', label='RVs From Keck Data')
    plt.plot(bjd, rv_model, 'or', label='Best Fit')
    plt.grid()
    plt.legend(fontsize=8)

    # Residuals
    frame2 = fig.add_axes((.1, .1, .8, .2))
    frame2.set_title('Residuals')
    frame2.set_yticks([-20, -10, 0, 10, 20])
    plt.plot(bjd, rv_resid, 'dr')
    plt.xlabel('BJD')
    plt.grid()

    if filename:
        plt.savefig(filename)
    if not silent:
        plt.show()
    plt.close('all')
