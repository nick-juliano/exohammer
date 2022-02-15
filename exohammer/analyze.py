#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:59:20 2021

@author: nickjuliano
"""

from astropy import constants as const
import matplotlib.pyplot as plt
from exohammer.utilities import ttvs

mearth = const.M_earth.cgs.value  # grams
msun = const.M_sun.cgs.value


def plot_ttvs(nplanets, measured, epoch, error, model, model_epoch, filename=None, silent=False):

	planet_designation = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
	fig, axes = plt.subplots(nplanets, figsize=(20, 16), sharex=True)
	fig.suptitle('ttvs', fontsize=30)
	ttv_model = ttvs(model, model_epoch)
	ttv_data = ttvs(measured, epoch)
	for i in range(nplanets):
		oc_data = ttv_data[i]
		oc_model = ttv_model[i]
		planet = planet_designation[i]
		ax = axes[i]
		ax.set_title('O-C Kepler 36-' + planet + ' with Fitted Orbital Elements',
		             fontsize=20)  # OPTIONAL: \n Burn In = 1,000, Iterations = 10,000', fontsize=20)
		ax.errorbar(epoch[i], oc_data, error[i], fmt='o', label='TTV From Measured Transits')
		ax.plot(model_epoch[i], oc_model, label='Best Fit (O-C)')
		ax.legend(fontsize=16)
	if filename != None:
		fig.savefig(filename)
	if silent != True:
		plt.show()
	plt.close('all')

def plot_rvs(bjd, rv, rv_err, rv_model, filename, silent=False):

	rv_resid = rv - rv_model

	fig1 = plt.figure(1)
	frame1 = fig1.add_axes((.1, .3, .8, .6))
	frame1.set_title('RVs')
	frame1.set_ylabel('')
	plt.errorbar(bjd, rv, rv_err, fmt='o', label='RVs From Keck Data')
	plt.plot(bjd, rv_model, 'or', label='best fit')  # Best fit model
	plt.ylabel('RV [m/s]')
	plt.xlabel('BJD')
	plt.grid()
	plt.legend(fontsize=8)

	# Residual plot
	frame2 = fig1.add_axes((.1, .1, .8, .2))
	plt.plot(bjd, rv_resid, 'dr')
	plt.grid()
	plt.xlabel('BJD')
	frame2.set_title('Residuals')
	frame2.set_yticks([-20, -10, 0, 10, 20])
	if filename != None:
		plt.savefig(filename)
	if silent != True:
		plt.show()
	plt.close('all')