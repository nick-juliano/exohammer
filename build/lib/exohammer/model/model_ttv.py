# -*- coding: utf-8 -*-

from ttvfast import ttvfast
from numpy import array, where, copy
from exohammer.utilities import generate_planets


def model_ttv(theta, system):
	dt = 0.4
	mstar = system.mstar
	epoch = system.epoch
	tmin = system.tmin - dt
	tmax = system.tmax + dt
	rvbjd = system.rvbjd
	nplanets = system.nplanets_fit
	planets = generate_planets(theta, system)
	rv_model = None

	model = ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)

	mod = []
	epo = []
	model_index, model_epoch, model_time, _, _, = model['positions']
	trim = min(where(array(model_time) == -2.))[0]
	model_index = array(model_index[:trim])
	model_epoch = array(model_epoch[:trim])
	model_time = array(model_time[:trim])
	for i in range(nplanets):
		idx = where(model_index == float(i))
		epoch_temp = copy(array(epoch[i]))
		epoch_temp = epoch_temp[epoch_temp <= max(model_epoch[idx])]
		model_temp = model_time[idx][epoch_temp]
		mod.append(model_temp.tolist())
		epo.append(epoch_temp.tolist())

	return mod, epo, rv_model
