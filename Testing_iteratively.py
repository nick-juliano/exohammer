#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:19:34 2021

@author: nickjuliano
"""

import exohammer as exo
from Input_Measurements import *
from Input_Measurements import rv_102021 as rv
import emcee

print(orbital_elements_4body)

kepler_36 = exo.planetary_system.planetary_system(2, 3, orbital_elements_4body, theta=None)
data = exo.data.data(mstar, [epoch, measured, error], rv, orbital_elements_4body)
run = exo.mcmc_run.mcmc_run(kepler_36, data)

run.explore_iteratively(total_iterations=10000000, checkpoints=1000, burnin_factor=.2, thinning_factor=.001,
	                    moves=[(emcee.moves.DEMove(), 0.5), (emcee.moves.DESnookerMove(), 0.5)],
	                    verbose=True, tune=True, silent=True)



