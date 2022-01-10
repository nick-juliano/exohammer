#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 09:32:20 2021

@author: nickjuliano
"""

import TTV_Module_Resid as TTV_Module

mstar = TTV_Module.Input_Measurements.mstar
ttv_data = [TTV_Module.Input_Measurements_w_resid.epoch,
            TTV_Module.Input_Measurements_w_resid.measured,
            TTV_Module.Input_Measurements_w_resid.error,
            TTV_Module.Input_Measurements_w_resid.ttv_holczer]

rv_data = TTV_Module.Input_Measurements_w_resid.rv_102021

# rv_resid = TTV_Module.Input_Measurements_w_resid.rv_resid
# rv_data[1]=rv_resid

orbital_elements = TTV_Module.Input_Measurements_w_resid.orbital_elements
# print(orbital_elements)
prob_functions = TTV_Module.prob_functions
# prob_functions = TTV_Module.prob_functions_separatedrvttv

data = TTV_Module.given.given(mstar,
                              ttv_data,  # [epoch, measured, error, given_ttvs],
                              rv_data,  # [bjd, mnvel, errvel],
                              orbital_elements)
# data.tmin=2459458.5-1
# data.tmax= 2459519.5+1
# import numpy as np
# data.rvbjd=np.linspace(2459458.5, 2459519.5, (519-458)*10).tolist()
kepler_36 = TTV_Module.planetary_system.planetary_system(0, orbital_elements, theta=None)
kepler_36.nplanets_ttv = 2
run = TTV_Module.mcmc_run.mcmc_run(kepler_36, data, prob_functions)

# run.plot_rvs(theta=kepler_36.fixed)
run.explore(niter=2000, thin_by=20)

TTV_Module.store.store_run(run)
run.summarize()

run.autocorr()
run.plot_chains()
run.plot_corner()
run.plot_ttvs()
run.plot_rvs()
run.plot_projected_rvs()
mod, epo, rv_model, rv_projected = TTV_Module.utilities.model_final_plot(run.theta_max, x=[data, kepler_36])
print(rv_projected)
