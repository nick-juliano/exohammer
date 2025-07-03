# -*- coding: utf-8 -*-
"""
Data Handling Module

Defines the `Data` class used to structure TTV and RV observations
for use in transit and radial velocity modeling.

Dependencies:
    - string
"""

import string


class Data:
    """
    Container class for observational data used in TTV and RV modeling.

    To be initialized and passed into a `System` instance.

    Attributes:
        m_star (float): Stellar mass in solar masses.
        epoch (list): List of epoch lists for each planet.
        measured (list): List of measured transit midpoints.
        error (list): Measurement uncertainties for TTVs.
        rvbjd (list): Barycentric Julian Dates of RV measurements.
        rvmnvel (list): Measured RV velocities.
        rverrvel (list): Uncertainties of RV measurements.
        ttvs (dict): Dictionary of TTV data by planet label.
        rvs (dict): Dictionary of RV data.
        tmin (float): Earliest time point in the combined dataset.
        tmax (float): Latest time point + 200-day buffer.
    """

    def __init__(self, m_star, ttvs=None, rvs=None):
        """
        Initialize a Data object with optional TTV and RV data.

        Args:
            m_star (float): Stellar mass in solar masses.
            ttvs (list, optional): Transit timing variation data:
                - ttvs[0] = list of epoch lists (one per planet)
                - ttvs[1] = list of measured midpoints
                - ttvs[2] = list of measurement errors
            rvs (list, optional): Radial velocity data:
                - rvs[0] = BJD list
                - rvs[1] = measured RVs
                - rvs[2] = RV errors
        """
        self.m_star = m_star
        self._times = []
        self._has_ttvs = False
        self._has_rvs = False
        self.ttvs = {}
        self.rvs = {}
        self.epoch = []
        self.measured = []
        self.error = []

        if rvs:
            self._has_rvs = True
            self.rvbjd = rvs[0]
            self.rvmnvel = rvs[1]
            self.rverrvel = rvs[2]
            self._times.extend(self.rvbjd)
            self._update_time_range()

        if ttvs:
            self._has_ttvs = True
            planet_labels = list(string.ascii_lowercase[1:len(ttvs[0]) + 1])  # starts from 'b'
            for i, label in enumerate(planet_labels):
                self.ttvs[label] = {
                    'epoch': ttvs[0][i],
                    'measured': ttvs[1][i],
                    'error': ttvs[2][i]
                }
                self.epoch.append(ttvs[0][i])
                self.measured.append(ttvs[1][i])
                self.error.append(ttvs[2][i])
                self._times.extend(ttvs[1][i])
            self._update_time_range()

    def add_ttv(self, planet_label, epoch, measured, error):
        """
        Add TTV data for a new planet.

        Args:
            planet_label (str): Identifier for the planet (e.g., 'b', 'c', etc.).
            epoch (list): Transit epochs.
            measured (list): Measured transit midpoints.
            error (list): Measurement uncertainties.
        """
        self.ttvs[planet_label] = {
            'epoch': epoch,
            'measured': measured,
            'error': error
        }
        self.epoch.append(epoch)
        self.measured.append(measured)
        self.error.append(error)
        self._has_ttvs = True
        self._times.extend(measured)
        self._update_time_range()

    def add_rvs(self, bjds, velocities, error):
        """
        Add RV data to the object.

        Args:
            bjds (list): Barycentric Julian Dates.
            velocities (list): Measured radial velocities.
            error (list): Measurement uncertainties.
        """
        self._has_rvs = True
        self.rvbjd = bjds
        self.rvmnvel = velocities
        self.rverrvel = error
        self._times.extend(bjds)
        self._update_time_range()

    def _update_time_range(self):
        """
        Update `tmin` and `tmax` based on the total time range in `_times`.

        Adds a 200-day buffer to `tmax` for safety.
        """
        if not self._times:
            self.tmin = None
            self.tmax = None
            return
        self.tmin = min(self._times)
        self.tmax = max(self._times) + 200  # generous buffer
