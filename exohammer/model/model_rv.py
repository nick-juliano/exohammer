# -*- coding: utf-8 -*-
"""
Radial Velocity Model Evaluation

Computes radial velocity predictions for a planetary system using `ttvfast`.

Dependencies:
    - ttvfast
    - numpy
    - exohammer.utilities.generate_planets
"""

from ttvfast import ttvfast
from numpy import array
from exohammer.utilities import generate_planets


def model_rv(theta, system):
    """
    Generate RV model predictions for a given system and parameter vector.

    Uses `ttvfast` to simulate the radial velocity curve based on Keplerian motion
    for a multi-planet system.

    Args:
        theta (list): Vector of sampled orbital parameters (e.g., from MCMC).
        system (System): Fully initialized `System` object containing star mass,
                         RV observation times, and orbital elements.

    Returns:
        tuple:
            mod (None): Placeholder to maintain consistent API with `model_both`.
            epo (None): Placeholder to maintain consistent API with `model_both`.
            rv_model (numpy.ndarray): Modeled radial velocities (in m/s).
    """
    dt = 0.4  # Time step for ttvfast integration
    m_star = system.m_star
    tmin = system.tmin - dt
    tmax = system.tmax + dt
    rvbjd = system.rvbjd
    au_per_day = 1731460  # AU/day to m/s conversion factor

    # Create planet objects from parameters
    planets = generate_planets(theta, system)

    # Run TTVFast for RVs
    model = ttvfast(planets, m_star, tmin, dt, tmax, rv_times=rvbjd)
    rv_model = array(model['rv']) * au_per_day

    return None, None, rv_model
