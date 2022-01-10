#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:32:49 2021

@author: nickjuliano
"""

import exohammer as exo
from Input_Measurements import *
from Input_Measurements import rv_102021 as rv
import numpy as np

run=exo.store.restore()
# run=exo.store.restore('/Users/nickjuliano/Documents/UNLV/Research/Kepler36/In_Development/exohammer_project/cherrycreekoutput/run_2021-12-29_13:58:05/pickle_2021-12-29_13:58:05.obj')
# run.theta_max=run.samples[np.argmax(run.sampler.get_log_prob(discard=int(1000000*0.2)))].tolist()
# planetary_system = exo.planetary_system.planetary_system(2, run.orbital_elements, theta=run.theta_max)
# data = exo.data.data(run.mstar, 
#                  [run.epoch, run.measured, run.error],              #[epoch, measured, error], 
#                  [run.rvbjd, run.rvmnvel, run.rverrvel],               #[bjd, mnvel, errvel], 
#                  run.orbital_elements)
# system = exo.system.system(planetary_system, data)

# mcmc_run       = exo.mcmc_run.mcmc_run(planetary_system, data)
# mcmc_run.discard=run.discard
# mcmc_run.samples=run.samples
# mcmc_run.output_path=run.output_path
# mcmc_run.sampler=run.sampler

# mod, epo, rv_model = exo.utilities.model(run.theta_max, system)

# #mcmc_run.plot_chains()
# samples=mcmc_run.sampler.get_chain(discard=100000, thin=200, flat=True)
# mcmc_run.samples=samples
# #mcmc_run.plot_chains()
# import matplotlib.pyplot as plt
# import corner
# figure = corner.corner(samples, labels=mcmc_run.system.variable_labels,quantiles=[0.16, 0.5, 0.84])
# plt.show()
# #mcmc_run.plot_chains()

# #print(run.samples[np.argmax(run.sampler.get_log_prob(discard=int(2000000*0.2)))])
# #exo.analyze.plot_rvs(run.rvbjd, run.rvmnvel, run.rverrvel, rv_model, filename=None)

# #exo.analyze.plot_ttvs(run.nplanets, run.measured, run.epoch, run.error, mod, epo)