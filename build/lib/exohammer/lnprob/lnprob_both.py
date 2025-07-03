# -*- coding: utf-8 -*-

from numpy import delete, all, array, inf, isfinite
from exohammer.utilities import trim, flatten_list



def lnprior(theta, system):
	flat = theta.copy().flatten()
	index = system.index
	minimum = system.theta_min
	maximum = system.theta_max

	(delete(flat, j) for j in index for i in range(len(flat), 0, -1) if i == j)

	lp = 0. if all(minimum < flat) and all(flat < maximum) else -inf

	gaus = theta[index]
	mu = system.mu
	sigma = system.sigma
	for i in range(len(index)):
		g = (((gaus[i] - mu[i]) / sigma[i]) ** 2.) * -.5
		lp += g

	return lp


def lnlike(theta, system):
	ttmodel, epo, rv_model = system.model(theta, system)
	sum_likelihood = 0

	# TTV
	comp, obs, err, ep = trim(system.nplanets_ttvs, system.epoch, system.measured, ttmodel, system.error, flatten=True)
	resid = array(obs) - array(comp)

	ttv_likelihood = ((array(resid) ** 2.) / (array(err) ** 2.) if len(resid) == len(err) else [-inf])

	for i in ttv_likelihood:
		sum_likelihood += i

	# RV
	rvresid = array(flatten_list(system.rvmnvel)) - (array(flatten_list(rv_model)))
	rv_likelihood = (array(rvresid) ** 2.) / (array(flatten_list(system.rverrvel)) ** 2.)

	for i in rv_likelihood:
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
