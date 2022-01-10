#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:19:34 2021

@author: nickjuliano
"""

import exohammer as exo
from Input_Measurements import *
from Input_Measurements import rv_102021 as rv

kepler_36 = exo.planetary_system.planetary_system(2, orbital_elements_3body, theta=None)
data = exo.data.data(mstar, [epoch, measured, error], rv, orbital_elements)
run = exo.mcmc_run.mcmc_run(kepler_36, data)

# lnprob=exo.prob_functions.lnprob(params, system)
# print(lnprob)
run.explore(100000, thin=1, verbose=True)

store = exo.store.store_run(run)
store.store()

run.plot_chains()
run.plot_rvs()
run.plot_ttvs()
run.autocorr()
run.summarize()
run.plot_corner()
