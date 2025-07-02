# -*- coding: utf-8 -*-
import string

class Data:
	"""
        Data Class

        To be initialized and passed into a `System` instance.
    """

	def __init__(self, m_star, ttvs=None, rvs=None):
		"""
		The Data Class

		Args:
			m_star (float): The mass of the host star in solar mass units
			ttvs (list): The ttv data such that `ttvs[0]=[epoch_b,..., epoch_i]`,
				`ttvs[1]=[measured_b,..., measured_i]`, `ttvs[2]=[error_b,..., error_i]`
			rvs (list): The rv data such that `rvs[0]=[bjd_b,..., bjd_i]`,
							`rvs[1]=[vel_b,..., vel_i]`, `rvs[2]=[error_b,..., error_i]`
		"""
		self.m_star = m_star
		self._times = []
		self._has_rvs = False
		self._has_ttvs = False
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
			for i in self.rvbjd:
				self._times.append(i)
			self._update_time_range()

		# self.nplanets_fit = len(self.epoch)
		if ttvs:
			planet_index = list(string.ascii_lowercase[1:len(ttvs)])
			self._has_ttvs = True
			self.ttvs = {}
			for i in range(len(planet_index)):
				planet_label = planet_index[i]
				self.ttvs[planet_label] = {}
				self.ttvs[planet_label]['epoch'] = ttvs[0][i]
				self.ttvs[planet_label]['measured'] = ttvs[1][i]
				self.ttvs[planet_label]['error'] = ttvs[2][i]
				self.epoch.append(self.ttvs[planet_label]['epoch'])
				self.measured.append(self.ttvs[planet_label]['measured'])
				self.error.append(self.ttvs[planet_label]['error'])
				for j in self.ttvs[planet_label]['measured']:
					self._times.append(j)
			self._update_time_range()
		return

	def add_ttv(self, planet_label, epoch, measured, error):
		"""
		add ttvs to the data class

		Args:
			epoch (list):
			measured (list): 
			error (list):
		"""
		self.ttvs[planet_label] = {}
		self.ttvs[planet_label]['epoch'] = epoch
		self.ttvs[planet_label]['measured'] = measured
		self.ttvs[planet_label]['error'] = error
		self.epoch.append(self.ttvs[planet_label]['epoch'])
		self.measured.append(self.ttvs[planet_label]['measured'])
		self.error.append(self.ttvs[planet_label]['error'])
		self._has_ttvs = True
		for j in self.ttvs[planet_label]['measured']:
			self._times.append(j)

		self._update_time_range()
		return
	
	def add_rvs(self, bjds, velocities, error):
		self._has_rvs = True
		self.rvbjd = bjds
		self.rvmnvel = velocities
		self.rverrvel = error
		for i in self.rvbjd:
			self._times.append(i)
		self._update_time_range()

	def _update_time_range(self):
		self.tmin = min(self._times)
		self.tmax = max(self._times)+200 # arbitrarily generous buffer 
		return