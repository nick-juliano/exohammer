# -*- coding: utf-8 -*-

from numpy import linspace, random
from exohammer.lnprob.initialize_prob import initialize_prob
from exohammer.model.initialize_model import initialize_model


class System:
    """
        System Class

        Primarily for organizing arguments to pass in to an `emcee.EnsembleSampler`
        instance. Kept available to the user for post-run analysis and serialization purposes.
    """
    def __init__(self, PlanetarySystem, Data):
        """

        Args:
            PlanetarySystem (object): A PlanetarySystem object
            Data (object): A Data object
        """
        self.orbital_elements = PlanetarySystem.orbital_elements
        self.nplanets_rvs = PlanetarySystem.nplanets_rvs
        self.nplanets_ttvs = PlanetarySystem.nplanets_ttvs  #
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

        self.mstar = Data.mstar
        self.epoch = Data.epoch
        self.measured = Data.measured
        self.error = Data.error
        self.rvbjd = Data.rvbjd
        self.rvmnvel = Data.rvmnvel
        self.rverrvel = Data.rverrvel
        self.tmin = Data.tmin
        self.tmax = Data.tmax
        self.model = initialize_model(self)
        self.lnprob = initialize_prob(self)

    def initial_state(self, nwalkers):

        initial_state = []
        orbital_elements = self.orbital_elements
        for i in range(nwalkers):
            p0 = []
            for j in orbital_elements:
                element = orbital_elements[j]
                if len(element) == 2:
                    minimum = element[0]
                    maximum = element[1]
                    options = linspace(minimum, maximum, 200)
                    p0.append(random.choice(options))
            initial_state.append(p0)
        return initial_state
