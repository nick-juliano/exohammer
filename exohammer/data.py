#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:01:22 2021

@author: nickjuliano
"""


class Data:
	"""
        Data Class

        To be initialized and passed into a `System` instance.
    """

	def __init__(self, mstar, ttvs, rvs):
		"""
		The Data Class

		Args:
			mstar (float): The mass of the host star in solar mass units
			ttvs (list): The ttv data such that `ttvs[0]=[epoch_b,..., epoch_i]`,
				`ttvs[1]=[measured_b,..., measured_i]`, `ttvs[2]=[error_b,..., error_i]`
			rvs (list): The rv data such that `rvs[0]=[bjd_b,..., bjd_i]`,
							`rvs[1]=[vel_b,..., vel_i]`, `rvs[2]=[error_b,..., error_i]`
		"""
		self.mstar = mstar
		self.epoch = ttvs[0]
		self.measured = ttvs[1]
		self.error = ttvs[2]
		self.rvbjd = rvs[0]
		self.rvmnvel = rvs[1]
		self.rverrvel = rvs[2]

		# self.nplanets_fit = len(self.epoch)

		if self.epoch is not None:
			x = self.measured

			tmin = min(x[0])
			tmax = max(x[0])

			for i in x:
				if min(i) < tmin:
					tmin = min(i)
				if max(i) > tmax:
					tmax = max(i)

			if self.rvbjd is not None:
				if min(self.rvbjd) < tmin:
					tmin = min(self.rvbjd)
				if max(self.rvbjd) > tmax:
					tmax = max(self.rvbjd)
			self.tmin = tmin
			self.tmax = tmax + 200

		elif self.epoch is None:
			if self.rvbjd is not None:
				x = self.rvbjd
				self.tmin = min(x)
				self.tmax = max(x)
