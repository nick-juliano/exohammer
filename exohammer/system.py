"""
System Initialization Module

Defines the `System` class, which organizes model and data attributes for
MCMC sampling using `emcee`. Also initializes model and log-probability functions.

Dependencies:
    numpy, exohammer.model.initialize_model, exohammer.lnprob.initialize_prob
"""

from numpy import linspace, random
from exohammer.lnprob.initialize_prob import initialize_prob
from exohammer.model.initialize_model import initialize_model


class System:
    """
    A container class for planetary system parameters and observational data.

    Used to organize the arguments and state passed to the `emcee.EnsembleSampler`
    and for storing MCMC run context (e.g., priors, likelihood functions, data).

    Attributes:
        orbital_elements (dict): Orbital element bounds for each parameter.
        nplanets_rvs (int): Number of planets with RV data.
        nplanets_ttvs (int): Number of planets with TTV data.
        p0 (list): Initial guess for sampling.
        ndim (int): Dimensionality of the parameter space.
        fixed (list): Fixed parameter values.
        theta_min (list): Minimum bounds for free parameters.
        theta_max (list): Maximum bounds for free parameters.
        variable_labels (list): Names of variable parameters.
        fixed_labels (list): Names of fixed parameters.
        sigma (list): Gaussian standard deviations for priors.
        mu (list): Gaussian means for priors.
        index (list): Index mapping of parameters.
        non_gaus (list): Labels for non-Gaussian priors.
        non_gaus_max (list): Upper bounds for non-Gaussian priors.
        non_gaus_min (list): Lower bounds for non-Gaussian priors.
        theta_ranges (list): Total allowed range for each parameter.

        m_star (float): Mass of the host star.
        epoch (list): Epochs of observed transits.
        measured (list): Measured transit times.
        error (list): Errors associated with measured times.
        rvbjd (list): Barycentric Julian Dates for RV observations.
        rvmnvel (list): Measured RV velocities.
        rverrvel (list): Uncertainties on RV velocities.
        tmin (float): Minimum time in the dataset.
        tmax (float): Maximum time in the dataset.

        model (callable): Initialized model function.
        lnprob (callable): Log-probability function for sampling.
    """

    def __init__(self, PlanetarySystem, Data):
        """
        Initialize the System with planetary parameters and observational data.

        Args:
            PlanetarySystem (object): Object containing orbital elements, priors, and model structure.
            Data (object): Object containing observational data (TTVs, RVs, etc.).
        """
        # Planetary parameters
        self.orbital_elements = PlanetarySystem.orbital_elements
        self.nplanets_rvs = PlanetarySystem.nplanets_rvs
        self.nplanets_ttvs = PlanetarySystem.nplanets_ttvs
        self.p0 = PlanetarySystem.p0
        self.ndim = PlanetarySystem.ndim
        self.fixed = PlanetarySystem.fixed
        self.theta_min = PlanetarySystem.theta_min
        self.theta_max = PlanetarySystem.theta_max
        self.variable_labels = PlanetarySystem.variable_labels
        self.fixed_labels = PlanetarySystem.fixed_labels
        self.sigma = PlanetarySystem.sigma
        self.mu = PlanetarySystem.mu
        self.index = PlanetarySystem.index
        self.non_gaus = PlanetarySystem.non_gaus
        self.non_gaus_max = PlanetarySystem.non_gaus_max
        self.non_gaus_min = PlanetarySystem.non_gaus_min
        self.theta_ranges = PlanetarySystem.theta_ranges

        # Observational data
        self.m_star = Data.m_star
        self.epoch = Data.epoch
        self.measured = Data.measured
        self.error = Data.error
        self.rvbjd = Data.rvbjd
        self.rvmnvel = Data.rvmnvel
        self.rverrvel = Data.rverrvel
        self.tmin = Data.tmin
        self.tmax = Data.tmax

        # Model and likelihood functions
        self.model = initialize_model(self)
        self.lnprob = initialize_prob(self)

    def initial_state(self, nwalkers):
        """
        Generate an initial ensemble state for MCMC walkers.

        This samples uniformly from within the bounds defined in
        `self.orbital_elements`.

        Args:
            nwalkers (int): Number of MCMC walkers.

        Returns:
            list: A list of `nwalkers` parameter vectors for initialization.
        """
        initial_state = []

        for _ in range(nwalkers):
            p0 = []
            for param, bounds in self.orbital_elements.items():
                if len(bounds) == 2:
                    minimum, maximum = bounds
                    options = linspace(minimum, maximum, 200)
                    p0.append(random.choice(options))
            initial_state.append(p0)

        return initial_state