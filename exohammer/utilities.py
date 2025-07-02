# -*- coding: utf-8 -*-

from astropy import constants as const
from numpy import log, argmax, array, zeros, sin, cos, pi, exp
from ttvfast import models

import matplotlib.pyplot as plt

mearth = const.M_earth.cgs.value  # grams
msun = const.M_sun.cgs.value
au_per_day = 1731460  # meters per second
mstar = 1.034


def sun_to_earth(solar_mass):
    """
    Converts Solar mass to Earth mass
    Args:
        solar_mass (float): solar mass value

    Returns:
        (float): Earth mass
    """
    earth_mass = solar_mass / msun * mearth
    return earth_mass


def best_fit(x, y):
    """

    Args:
        x (list): x values
        y (list): y values

    Returns:
        (tuple): Tuple of slope and intercept.
    """
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    n = len(x)  # or len(Y)
    numer = sum([xi * yi for xi, yi in zip(x, y)]) - n * xbar * ybar
    denum = sum([xi ** 2 for xi in x]) - n * xbar ** 2
    m = numer / denum
    b = ybar - m * xbar
    return m, b


def ttvs(measured, epoch):
    """

    Args:
        measured (list): transit mid-points
        epoch (list): epoch of each transit mid-point

    Returns:
        (list): transit timing variations
    """
    oc = []
    for i in range(len(measured)):
        slope, inter = best_fit(epoch[i], measured[i])
        y = inter + slope * array(epoch[i])
        oc.append((measured[i] - y))
    return oc


def trim(nplanets, epoch, measured, model, error, flatten=True):
    """

    Args:
        nplanets (int): The number of planets
        epoch (list): A list of epochs
        measured (list): A list of measured values
        model (list): A list of modeled values
        error (list): A list of errors
        flatten (bool): whether to return the list as flattened numpy arrays or multidimensional lists

    Returns:
        (tuple): model, measured, error, epoch lists trimmed to the shortest list length
    """
    mod = []
    meas = []
    err = []
    ep = []
    for i in range(nplanets):
        if len(measured[i]) > len(model[i]):
            meas.append(measured[i][:len(model[i])])
            mod.append(model[i])
            err.append(error[i][:len(epoch[i])])
            ep.append(epoch[i])
        elif len(model[i]) > len(measured[i]):
            meas.append(measured[i])
            mod.append(model[i][:len(epoch[i])])
            err.append(error[i])
            ep.append(epoch[i])
        elif len(model[i]) == len(measured[i]):
            meas.append(measured[i])
            mod.append(model[i])
            err.append(error[i])
            ep.append(epoch[i])
    if flatten:
        mod = array(flatten_list(mod))
        meas = array(flatten_list(meas))
        err = array(flatten_list(err))
        ep = array(flatten_list(ep))
    return mod, meas, err, ep


def flatten_list(_2d_list):
    """
    Converts 2-dimensional list to 1-dimensional list

    Args:
        _2d_list (list): list

    Returns:
        (list): flattened list
    """
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def generate_planets(theta, system):
    nplanets = system.nplanets_rvs
    planet_labels = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    element_keys = ['mass', 'period', 'eccentricity', 'inclination',
                    'longnode', 'argument', 'mean_anomaly']

    # Combine fixed and variable parameters
    orb_elements = dict(zip(system.fixed_labels, system.fixed))
    orb_elements.update(zip(system.variable_labels, theta))

    planets = []
    for j in range(nplanets):
        label = planet_labels[j]
        params = {key: orb_elements[f"{key}_{label}"] for key in element_keys}
        planet = models.Planet(
            mass=params['mass'],
            period=params['period'],
            eccentricity=params['eccentricity'],
            inclination=params['inclination'],
            longnode=params['longnode'],
            argument=params['argument'],
            mean_anomaly=params['mean_anomaly']
        )
        planets.append(planet)

    return planets


def plot_periodogram(t, rv, title):
    """
    Plots the periodogram of radial velocity measurements

    Args:
        t (list): The list of times
        rv (list): The radial velocity values at each time
        title (str): The title of the plot
    """
    t = array(t)
    t = t - t[0]
    rv = array(rv)
    s1 = len(t)
    sx = sum(rv) / len(rv)
    sx2 = 0.0
    for m in range(s1):
        sx2 = (rv[m] - sx) ** 2 + sx2
    sx2 = sx2 / s1
    pstart = 60
    pstop = t[-1]
    pint = 1
    ni = -6.362 + 1.193 * s1 + 0.00098 * s1 ** 2
    pw = zeros([int((pstop - pstart) / pint), 3])
    for m in range(int((pstop - pstart) / pint)):
        pt = pstart + (m + 1) * pint
        w = 2 * pi / pt
        tao = 0.0
        p1 = sum(rv * cos(w * (t - tao)))
        p3 = sum(rv * sin(w * (t - tao)))
        p2 = sum((cos(w * (t - tao))) ** 2)
        p4 = sum((sin(w * (t - tao))) ** 2)
        px = 0.5 * (p1 ** 2 / p2 + p3 ** 2 / p4) / sx2
        pw[m, 0] = pt
        pw[m, 1] = px
        pw[m, 2] = 1 - (1 - exp(-px)) ** ni
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l1 = ax.plot(pw[:, 0], pw[:, 1], 'b', alpha=0.9, label='Power')
    ax.set_xlim([60, 700])
    ax.set_xlabel('Period (day)', fontsize=16)

    ax2 = ax.twinx()
    l2 = ax2.plot(pw[:, 0], pw[:, 2], 'y', alpha=0.9, label='False Alarm Probability')
    l_both = l1 + l2
    labs = [x.get_label() for x in l_both]
    ax.legend(l_both, labs, bbox_to_anchor=(0.78, 0.9))
    ax.set_title(title)
    ax.set_ylabel('Power', fontsize=16)
    ax.set_xticks([90, 180, 360, 540])
    ax2.set_ylabel('False Alarm Probability', fontsize=16)
    plt.close('all')


def sampler_to_theta_max(run):
    """
    Computes the theta value with the highest lnprob. Both returns the value and adds it to the run object

    Args:
        run (object): An explored MCMC run

    Returns:
        list: The theta value with the highest lnprob
    """
    samples = run.sampler.get_chain(flat=True, thin=run.thin)
    run.theta_max = samples[argmax(run.sampler.get_log_prob(flat=True, thin=run.thin))]
    return run.theta_max

def bic(run):
    """
    Computes the Bayesian Information Criterion of an MCMC run. Both returns the value and adds it to the run object

    Args:
        run (object): An explored MCMC run

    Returns:
        float: The bayesian information criterion
    """
    k = len(run.theta_max)
    n = 0
    for i in range(run.system.nplanets_ttvs):
        n += len(run.system.measured[i])
    n += len(run.system.rvbjd)
    lnprob = argmax(run.sampler.get_log_prob(flat=True, thin=run.thin))
    bayesian_information_criterion = k * log(n) - (2 * lnprob)
    run.bic = bayesian_information_criterion
    return bayesian_information_criterion


def plot_ttvs(nplanets, measured, epoch, error, model, model_epoch, filename=None, silent=False):
    """
    Plot the ttvs

    Args:
        nplanets (int): The number of planets
        measured (list): The list of measured ttvs
        epoch (list): The epochs of the measured ttvs
        error (list): The errors of the measured ttvs
        model (list): The list of modeled ttvs
        model_epoch (list): The epochs of the modeled ttvs
        filename (str): Where to save the output files
        silent (bool): Specifies if the plot should be rendered in the IDE or simply saved
    """
    planet_designation = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    fig, axes = plt.subplots(nplanets, figsize=(20, 16), sharex=True)
    fig.suptitle('ttvs', fontsize=30)
    ttv_model = ttvs(model, model_epoch)
    ttv_data = ttvs(measured, epoch)
    for i in range(nplanets):
        oc_data = ttv_data[i]
        oc_model = ttv_model[i]
        planet = planet_designation[i]
        ax = axes[i]
        ax.set_title('O-C Kepler 36-' + planet + ' with Fitted Orbital Elements', fontsize=20)
        ax.errorbar(epoch[i], oc_data, error[i], fmt='o', label='TTV From Measured Transits')
        ax.plot(model_epoch[i], oc_model, label='Best Fit (O-C)')
        ax.legend(fontsize=16)
    if filename:
        fig.savefig(filename)
    if not silent:
        plt.show()
    plt.close('all')


def plot_rvs(bjd, rv, rv_err, rv_model, filename, silent = False):
    """
    Plot radial velocities

    Args:
        bjd (list): The time of the rv measurements
        rv (list): The measured rvs
        rv_err (list): The measured rv errors
        rv_model (list): The modeled rvs
        filename (str): Where to save the files
        silent (bool): Specifies if the plot should be rendered in the IDE or simply saved
    """
    rv_resid = rv - rv_model
    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((.1, .3, .8, .6))
    frame1.set_title('RVs')
    frame1.set_ylabel('')
    plt.errorbar(bjd, rv, rv_err, fmt='o', label='RVs From Keck Data')
    plt.plot(bjd, rv_model, 'or', label='best fit')  # Best fit model
    plt.ylabel('RV [m/s]')
    plt.xlabel('BJD')
    plt.grid()
    plt.legend(fontsize=8)

    # plot residuals
    frame2 = fig1.add_axes((.1, .1, .8, .2))
    plt.plot(bjd, rv_resid, 'dr')
    plt.grid()
    plt.xlabel('BJD')
    frame2.set_title('Residuals')
    frame2.set_yticks([-20, -10, 0, 10, 20])
    if filename:
        plt.savefig(filename)
    if not silent:
        plt.show()
    plt.close('all')

