import ttvfast
import numpy as np

from exohammer.utilities import generate_planets


def model_rv(theta, system):
	dt = 0.4
	mstar = system.mstar
	tmin = system.tmin - dt
	tmax = system.tmax + dt
	rvbjd = system.rvbjd
	planets = generate_planets(theta, system)
	au_per_day = 1731460

	mod = None
	epo = None

	model = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)

	rv_model = model['rv']
	rv_model = np.array(rv_model) * au_per_day

	return mod, epo, rv_model
