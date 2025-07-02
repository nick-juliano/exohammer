# -*- coding: utf-8 -*-

from numpy import linspace, random

class PlanetarySystem:
    """
        PlanetarySystem Class

        To be initialized and passed into a `System` instance.
    """
    def __init__(self, nplanets_ttvs, nplanets_rvs, orbital_elements, theta=None):
        """

        Args:
            nplanets_ttvs (int): The number of planets to fit to the ttv data
            nplanets_rvs (int): The number of planets to fit to the rv data
            orbital_elements (dict): A dictionary of the orbital elements of each planet.
                Please see the README for advice on the structure of this argument
            theta (list): A list of orbital elements (typically generated within an MCMC run)
                It is recommended to only define theta here if you want to evaluate previously fitted results
        """
        self.orbital_elements = orbital_elements
        self.nplanets_rvs = nplanets_rvs
        self.nplanets_ttvs = nplanets_ttvs

        if type(theta) == list:
            self.theta = theta

        orbital_elements = self.orbital_elements
        p0 = []
        fixed = []
        theta_min = []
        theta_max = []
        variable_labels = []
        fixed_labels = []
        sigma = []
        mu = []
        index = []
        non_gaus = []
        non_gaus_max = []
        non_gaus_min = []
        theta_ranges = []
        k = 0

        for i in orbital_elements:
            element = orbital_elements[i]
            # List of length 1 == fixed param
            if len(element) == 1:
                fixed_labels.append(str(i))
                k += 1
                fixed.append(element[0])
                non_gaus.append(element[0])
                non_gaus_max.append(element[0] + element[0] / 1.e9)
                non_gaus_min.append(element[0] - element[0] / 1.e9)
            # List of length 2 == flat params
            if len(element) == 2:
                minimum = element[0]
                maximum = element[1]
                options = linspace(minimum, maximum, 200)
                theta_ranges.append(options)
                theta_min.append(minimum)
                theta_max.append(maximum)
                p0.append(random.choice(options))
                variable_labels.append(str(i))
                non_gaus.append(element)
                non_gaus_max.append(maximum)
                non_gaus_min.append(minimum)
            # List of length 3 == gaussian params
            if len(element) == 3:
                index.append((i - k) + (len(orbital_elements)))
                mu.append(element[0])
                sigma.append(element[1])

        self.p0 = p0
        self.ndim = len(p0)
        self.fixed = fixed
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.variable_labels = variable_labels
        self.fixed_labels = fixed_labels
        self.sigma = sigma
        self.mu = mu
        self.index = index
        self.non_gaus = non_gaus
        self.non_gaus_max = non_gaus_max
        self.non_gaus_min = non_gaus_min
        self.theta_ranges = theta_ranges
