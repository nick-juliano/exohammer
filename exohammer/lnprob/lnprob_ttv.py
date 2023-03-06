# -*- coding: utf-8 -*-

from numpy import delete, all, array, inf, isfinite
from exohammer.utilities import trim, flatten_list

def lnprior(theta, system):
	flat = theta.copy().flatten()
	index = system.index
	minimum = system.theta_min
	maximum = system.theta_max
	mu = system.mu
	sigma = system.sigma
	for i in range(len(flat), 0, -1):
		for j in index:
			if i == j:
				flat = delete(flat, j)
	lp = 0. if all(minimum < flat) and all(flat < maximum) else -inf
	gaus = theta[index]
	for i in range(len(index)):
		g = (((gaus[i] - mu[i]) / sigma[i]) ** 2.) * -.5
		lp += g
	return lp


def lnlike(theta, system):
	ttmodel, epo, rv_model = system.model(theta, system)
	sum_likelihood = 0

	mod, meas, err, ep = trim(system.nplanets_ttvs, system.epoch, system.measured, ttmodel, system.error,
	                          flatten=False)
	obs = []
	comp = []
	nplanets_ttvs = system.nplanets_ttvs
	for i in range(nplanets_ttvs):
		obs.append(meas[i])
		comp.append(mod[i])
	obs = flatten_list(obs)
	comp = flatten_list(comp)
	err = flatten_list(err)

	resid = array(obs) - array(comp)

	if len(resid) == len(err):
		ttv_likelihood = (array(resid) ** 2.) / (array(err) ** 2.)

	else:
		ttv_likelihood = [-inf]

	for i in ttv_likelihood:
		sum_likelihood += i

	likelihood = -0.5 * sum_likelihood

	if not isfinite(likelihood):
		likelihood = -inf
	return likelihood


def lnprob(theta, system):
	lp = lnprior(theta, system)
	if not isfinite(lp):
		return -inf
	else:
		return lp + lnlike(theta, system)
