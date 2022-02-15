#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:19:34 2021

@author: nickjuliano
"""

import exohammer as exo
from Input_Measurements import *
from Input_Measurements import rv_102021 as rv

kepler_36 = exo.planetary_system.planetary_system(2, 3, orbital_elements_4body, theta=None)
data = exo.data.data(mstar, [epoch, measured, error], rv, orbital_elements_4body)
run = exo.mcmc_run.mcmc_run(kepler_36, data)

# lnprob=exo.prob_functions.lnprob(params, system)
# print(lnprob)
niter_total=10000000
chopper=1000
chopped=int(niter_total/chopper)
run.explore(10000, thin=10, verbose=True)
run.plot_chains()
run.plot_rvs()
run.plot_ttvs()
run.autocorr()
run.plot_corner()
for i in range(chopper):
    print(str(i))
    run.explore_again(chopped)
    run.thin=int(chopped/20)
    if i%5==0:

        re = exo.store.store_run(run)
        re.store()
        run.plot_chains()
        run.plot_rvs()
        run.plot_ttvs()
        run.autocorr()
        run.summarize()
        run.plot_corner()
print(run.theta_max)

store = exo.store.store_run(run)
store.store()

run.plot_chains()
run.plot_rvs()
run.plot_ttvs()
run.autocorr()
run.summarize()
run.plot_corner()

