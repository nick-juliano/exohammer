from exohammer.utilities import flatten_list
import numpy as np


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
				flat = np.delete(flat, j)
	lp = 0. if np.all(minimum < flat) and np.all(flat < maximum) else -np.inf
	gaus = theta[index]
	for i in range(len(index)):
		g = (((gaus[i] - mu[i]) / sigma[i]) ** 2.) * -.5
		lp += g
	return lp


def lnlike(theta, system):
	ttmodel, epo, rv_model = system.model(theta, system)
	sum_likelihood = 0

	# RV
	rvresid = np.array(flatten_list(system.rvmnvel)) - (np.array(flatten_list(rv_model)))
	rv_likelihood = (np.array(rvresid) ** 2.)

	for i in rv_likelihood:
		sum_likelihood += i

	likelihood = -0.5 * sum_likelihood
	if not np.isfinite(likelihood):
		likelihood = -np.inf
	return likelihood


def lnprob(theta, system):
	lp = lnprior(theta, system)
	if not np.isfinite(lp):
		return -np.inf
	else:
		return lp + lnlike(theta, system)
