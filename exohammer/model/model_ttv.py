# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Transit Timing Variation (TTV) Model Evaluation

Computes modeled transit midpoints for a multi-planet system using `ttvfast`.

Dependencies:
    - ttvfast
    - numpy
    - exohammer.utilities.generate_planets
"""

from ttvfast import ttvfast
from numpy import array, where, copy
from exohammer.utilities import generate_planets


def model_ttv(theta, system):
    """
    Generate transit timing variation (TTV) predictions for a system.

    Uses `ttvfast` to simulate mid-transit times for each planet in the system.

    Args:
        theta (list): Vector of orbital parameters for all modeled planets.
        system (System): Fully initialized `System` object containing planetary
                         data, epochs, and stellar mass.

    Returns:
        tuple:
            mod (list of lists): Modeled transit midpoints per planet.
            epo (list of lists): Epochs corresponding to each modeled transit.
            rv_model (None): Placeholder to maintain API consistency with `model_both()`.
    """
    dt = 0.4
    m_star = system.m_star
    tmin = system.tmin - dt
    tmax = system.tmax + dt
    rvbjd = system.rvbjd
    epoch = system.epoch
    nplanets = system.nplanets_fit  # typically equals nplanets_ttvs
    planets = generate_planets(theta, system)

    # Run TTVFast simulation
    model = ttvfast(planets, m_star, tmin, dt, tmax, rv_times=rvbjd)

    mod = []  # List of modeled midpoints
    epo = []  # List of used epochs
    rv_model = None  # Placeholder for consistency

    model_index, model_epoch, model_time, _, _ = model['positions']

    # Remove sentinel value (-2.0) padding
    sentinel_indices = where(array(model_time) == -2.)[0]
    trim = sentinel_indices[0] if len(sentinel_indices) > 0 else len(model_time)

    model_index = array(model_index[:trim])
    model_epoch = array(model_epoch[:trim])
    model_time = array(model_time[:trim])

    # Extract transit times for each planet
    for i in range(nplanets):
        idx = where(model_index == float(i))
        epoch_temp = copy(array(epoch[i]))
        valid_epochs = epoch_temp[epoch_temp <= max(model_epoch[idx])]
        model_temp = model_time[idx][valid_epochs]
        mod.append(model_temp.tolist())
        epo.append(valid_epochs.tolist())

    return mod, epo, rv_model
