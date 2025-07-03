"""
Planetary System Configuration Module

Defines the `PlanetarySystem` class, which stores parameter definitions and
prior structures for each planet to be used in transit timing variation (TTV)
and radial velocity (RV) modeling.

Supports fixed values, uniform priors (2-element lists), and Gaussian priors (3-element lists).
"""

from numpy import linspace, random


class PlanetarySystem:
    """
    Defines the planetary system's orbital elements, priors, and labeling.

    To be passed into a `System` instance to initialize the modeling pipeline.

    Supports three types of orbital element definitions:
        - [val]: fixed parameter
        - [min, max]: uniform prior
        - [mu, sigma, _]: Gaussian prior (third value is ignored, used for compatibility)

    Attributes:
        orbital_elements (dict): User-specified parameter definitions.
        nplanets_ttvs (int): Number of planets with TTV data.
        nplanets_rvs (int): Number of planets with RV data.
        theta (list): Optional vector of parameters for evaluating previous fits.

        p0 (list): Initial parameter vector (random from uniform priors).
        ndim (int): Number of free parameters.
        fixed (list): Fixed parameters (not sampled).
        theta_min (list): Lower bounds of free parameters.
        theta_max (list): Upper bounds of free parameters.
        variable_labels (list): Labels for free parameters.
        fixed_labels (list): Labels for fixed parameters.
        sigma (list): Standard deviations for Gaussian priors.
        mu (list): Means for Gaussian priors.
        index (list): Indices used to align Gaussian priors.
        non_gaus (list): All non-Gaussian priors (uniform or fixed).
        non_gaus_max (list): Max bounds for non-Gaussian priors.
        non_gaus_min (list): Min bounds for non-Gaussian priors.
        theta_ranges (list): Discretized value ranges for each free parameter.
    """

    def __init__(self, nplanets_ttvs, nplanets_rvs, orbital_elements, theta=None):
        """
        Initialize a PlanetarySystem with TTV/RV counts and orbital element priors.

        Args:
            nplanets_ttvs (int): Number of planets using TTV observations.
            nplanets_rvs (int): Number of planets using RV observations.
            orbital_elements (dict): Dictionary where keys are parameter labels and values are:
                - [value] for fixed
                - [min, max] for uniform prior
                - [mu, sigma, dummy] for Gaussian prior
            theta (list, optional): Previously fit parameter vector, for evaluation purposes.
        """
        self.orbital_elements = orbital_elements
        self.nplanets_rvs = nplanets_rvs
        self.nplanets_ttvs = nplanets_ttvs
        self.theta = theta if isinstance(theta, list) else None

        # Internal configuration lists
        p0 = []                  # Starting parameters
        fixed = []              # Fixed parameter values
        theta_min = []          # Lower bounds
        theta_max = []          # Upper bounds
        variable_labels = []    # Labels for sampled params
        fixed_labels = []       # Labels for fixed params
        sigma = []              # Stddev for Gaussian priors
        mu = []                 # Mean for Gaussian priors
        index = []              # Index mapping for Gaussian priors
        non_gaus = []           # Flat and fixed priors
        non_gaus_max = []       # Max values for flat/fixed
        non_gaus_min = []       # Min values for flat/fixed
        theta_ranges = []       # Discretized ranges (200 pts each)

        k = 0  # Offset counter for fixed parameters (used in Gaussian index mapping)

        for key in orbital_elements:
            element = orbital_elements[key]

            # Fixed parameter
            if len(element) == 1:
                fixed_labels.append(str(key))
                val = element[0]
                fixed.append(val)
                non_gaus.append(val)
                non_gaus_max.append(val + val / 1e9)
                non_gaus_min.append(val - val / 1e9)
                k += 1

            # Uniform prior (flat)
            elif len(element) == 2:
                minimum, maximum = element
                options = linspace(minimum, maximum, 200)
                theta_ranges.append(options)
                theta_min.append(minimum)
                theta_max.append(maximum)
                p0.append(random.choice(options))
                variable_labels.append(str(key))
                non_gaus.append(element)
                non_gaus_max.append(maximum)
                non_gaus_min.append(minimum)

            # Gaussian prior
            elif len(element) == 3:
                mu.append(element[0])
                sigma.append(element[1])
                # Indexing adjusted to place Gaussian params after non-Gaussian
                index.append((len(p0) - k) + len(orbital_elements))

        # Store attributes
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
