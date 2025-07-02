# -*- coding: utf-8 -*-

from ttvfast import ttvfast
from numpy import array
from exohammer.utilities import generate_planets


def model_rv(theta, system):
	dt = 0.4
	m_star = system.m_star
	tmin = system.tmin - dt
	tmax = system.tmax + dt
	rvbjd = system.rvbjd
	planets = generate_planets(theta, system)
	au_per_day = 1731460

	mod = None
	epo = None

	model = ttvfast(planets, m_star, tmin, dt, tmax, rv_times=rvbjd)

	rv_model = model['rv']
	rv_model = array(rv_model) * au_per_day

	return mod, epo, rv_model
