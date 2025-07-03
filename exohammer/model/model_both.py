# -*- coding: utf-8 -*-
"""
TTV and RV Model Evaluation

This module defines `model_both`, which computes both TTV and RV models
from orbital parameters using `ttvfast`.

Dependencies:
    - ttvfast
    - numpy
    - exohammer.utilities.generate_planets
"""

from ttvfast import ttvfast
from numpy import array, where, copy
from exohammer.utilities import generate_planets, au_per_day


def model_both(theta, system):
    """
    Generate transit and radial velocity model predictions.

    Uses `ttvfast` to simulate transit midpoints and radial velocities
    for a planetary system given a vector of orbital parameters.

    Args:
        theta (list): Parameter vector (typically sampled in MCMC).
        system (System): A fully initialized System object, including planetary
                         parameters, stellar mass, and observational epochs.

    Returns:
        tuple:
            mod (list of lists): Modeled transit midpoints for each planet.
            epo (list of lists): Corresponding epochs used for each model series.
            rv_model (array): Radial velocity predictions (in m/s).
    """
    dt = 0.3  # integration timestep (days)

    # Generate planet objects and run TTVFast
    planets = generate_planets(theta, system)
    model = ttvfast(planets, system.m_star,
                    system.tmin - dt, dt, system.tmax + dt,
                    rv_times=system.rvbjd)

    # Convert RVs from AU/day to m/s
    rv_model = array(model['rv']) * au_per_day

    mod = []  # TTV results
    epo = []  # Epochs used
    model_index, model_epoch, model_time, _, _ = model['positions']

    # Remove sentinel value (-2.) padding in output
    sentinel_indices = where(model_time == -2.)[0]
    trim = sentinel_indices[0] if len(sentinel_indices) > 0 else len(model_time)
    model_index = array(model_index[:trim])
    model_epoch = array(model_epoch[:trim])
    model_time = array(model_time[:trim])

    # Extract transit model for each TTV planet
    for i in range(system.nplanets_ttvs):
        idx = where(model_index == float(i))
        epoch_temp = copy(array(system.epoch[i]))
        valid_epochs = epoch_temp[epoch_temp <= max(model_epoch[idx])]
        model_temp = model_time[idx][valid_epochs]
        mod.append(model_temp.tolist())
        epo.append(valid_epochs.tolist())

    return mod, epo, rv_model
