#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:04:17 2021

@author: nickjuliano
"""


class mcmc_run:
	"""
    **mcmc_run** class

    The primary object of *exohammer*. Once all other defining values have been passed into this class, the mcmc_run object can then perform the Bayesian analysis of the the system.

    """
	import emcee

	def __init__(self, planetary_system, data, lnprob=None):
		"""
        Initialize **mcmc_run** object

        Parameters
        ----------
        planetary_system : object
            exohammer.planetary_system.planetary_system object
        data : object
            exohammer.data.data object
        lnprob : function
            probability function. Must be of the format lnprob(theta, args). Optional (determined by initialize_prob.py if None)
        """
		import os
		import datetime
		import emcee
		from exohammer.system import system

		date = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time())[:-7]
		output_path = str(os.path.abspath(os.getcwd()) + "/Output/")
		testdir=output_path + 'run_' + date
		if not os.path.exists(testdir):
			os.makedirs(testdir)
		output_path = output_path + 'run_' + date + "/"
		self.output_path = output_path
		self.date = date
		self.EnsembleSampler = emcee.EnsembleSampler
		self.system = system(planetary_system, data, lnprob)

	def explore(self, iterations, thin=1, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
	            verbose=True, tune=True):
		"""
        Advances the ensemble

        Parameters
        ----------
        iterations : int
            The amount of steps to propose
        thin : int
            Use only every thin steps from the chain (default: 1)
        moves : object
            This can be a single move object, a list of moves,
            or a “weighted” list of the form [(emcee.moves.StretchMove(), 0.1), ...].
            When running, the sampler will randomly select a move from this list
            (optionally with weights) for each proposal.
            (default: [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        verbose : bool
            If True, a progress bar will be shown as the sampler progresses. (default: True)
        tune : bool
            If True, the parameters of some moves will be automatically tuned. (default: True)
        """
		import numpy as np
		walk = self.system.ndim * 3
		self.nwalkers = int(walk) if ((int(walk) % 2) == 0) else int(walk) + 1
		self.nburnin = int(0.1 * iterations)
		self.thin = thin
		self.niter = int(iterations - self.nburnin)
		self.moves = moves
		self.discard = 0
		self.tune=tune
		lnprob = self.system.lnprob
		ndim = self.system.ndim
		nwalkers = self.nwalkers
		nburnin = self.nburnin
		thin = self.thin

		p0 = self.system.initial_state(self.nwalkers)
		# Initialize the sampler
		sampler = self.EnsembleSampler(self.nwalkers, ndim, lnprob, args=[self.system],
		                               moves=moves)  # , backend=backend)

		print("Running burn-in... " + str(self.nburnin) + " steps")

		p0, _, _ = sampler.run_mcmc(p0, nburnin, progress=verbose)
		sampler.reset()

		print("Running production..." + str(self.niter) + " steps")
		pos, prob, state = sampler.run_mcmc(p0, self.niter, thin=thin, progress=verbose, tune=tune)

		self.sampler = sampler
		self.pos = pos
		self.prob = prob
		self.state = state

		def sampler_to_theta_max(self):
			import numpy as np
			sampler = self.sampler
			samples = sampler.get_chain(flat=True, thin=self.thin)
			self.samples = samples
			self.theta_max = samples[np.argmax(sampler.get_log_prob(flat=True, thin=self.thin))]

		sampler_to_theta_max(self)

	def explore_again(self, niter, verbose=True):
		import numpy as np
		print("Running production..." + str(niter) + " steps")
		self.niter+=niter
		pos, prob, state = self.sampler.run_mcmc(self.pos, niter, thin=self.thin, progress=verbose, tune=self.tune)
		self.pos=pos #np.append(self.pos, pos.reshape(len(pos),len(pos[-1])), axis=0)
		self.prob=prob
		self.state=state

		def sampler_to_theta_max(self):
			import numpy as np
			sampler = self.sampler
			samples = sampler.get_chain(flat=True, thin=self.thin)
			self.samples = samples
			self.theta_max = samples[np.argmax(sampler.get_log_prob(flat=True, thin=self.thin))]

		sampler_to_theta_max(self)

	def plot_corner(self, samples=None, save=True):
		"""
        Generates a corner plot and optionally saves it to the output path.

        Parameters
        ----------
        samples : list
            Samples from an mcmc_run.explore() run.
            It is recommended to keep samples=None unless you are
            attempting to plot a poorly-pickled previous run.
        save : bool
            If True, saves the corner plot to the run's output path.
		"""
		import matplotlib.pyplot as plt
		import corner
		filename = self.output_path + "corner_" + self.date + '.png'
		if samples==None:
			samples = self.samples
		figure = corner.corner(samples, labels=self.system.variable_labels)
		if save == True:
			figure.savefig(filename)
		plt.show()

	def plot_chains(self, discard=None, samples=None, save=True):
		"""
        Generates a chain plot and optionally saves it to the output path.

        Parameters
        ----------
        discard : int
            The number of steps at the beginning of the run to omit from the plot (optional)
        samples : list
            Samples from an mcmc_run.explore() run.
            It is recommended to keep samples=None unless you are
            attempting to plot a poorly-pickled previous run.
        save : bool
            If True, saves the plot to the run's output path.
		"""
		import matplotlib.pyplot as plt
		filename = self.output_path + "Chains_" + self.date + '.png'
		# samples = self.samples
		if discard == None:
			discard = self.discard
		if samples == None:
			samples = self.sampler.get_chain(discard=discard)
		fig, axes = plt.subplots(len(self.system.variable_labels), figsize=(20, 30), sharex=True)
		fig.suptitle('chains', fontsize=30)
		for i in range(len(self.system.variable_labels)):
			ax = axes[i]
			ax.plot(samples[:, :, i], "k", alpha=0.3)
			ax.set_xlim(0, len(samples))
			ax.set_ylabel(self.system.variable_labels[i])
			ax.yaxis.set_label_coords(-0.1, 0.5)
		if save == True:
			plt.savefig(filename)
		plt.show()

	def summarize(self):
		space = """

            """
		summary = """
            This run sampled %i walkers with %i iterations with a burn-in of %i.  

            The chain was thinned by %i

            The resultant orbital elements are below:

            """ % (self.nwalkers, self.niter, self.discard, self.thin)
		run_description = open(self.output_path + "/run_description_" + self.date + ".txt", "w+")
		run_description.write(summary)
		run_description.write(space)
		run_description.write(str(self.system.variable_labels))
		run_description.write(space)
		# theta=self.theta_max
		run_description.write(str(self.theta_max))
		print(summary)
		print(self.system.variable_labels)
		print(self.theta_max)

	def autocorr(self):
		filename = self.output_path + "autocor_" + self.date + '.txt'
		# autocor  = open(self.output_path + "/autocor_"+self.date+".txt", "w+")
		try:
			tau = self.sampler.get_autocorr_time()
		except Exception as e:
			print(str(e))
			tau = e
		with open(filename, 'w') as f:
			f.write(str(tau))
		return tau

	def plot_rvs(self):
		from exohammer.utilities import model
		from exohammer.analyze import plot_rvs

		__, __, rv_model = model(self.theta_max, self.system)
		filename = self.output_path + "rvs_" + self.date + '.png'
		plot_rvs(self.system.rvbjd, self.system.rvmnvel, self.system.rverrvel, rv_model, filename)

	def plot_ttvs(self):
		from exohammer.utilities import model
		from exohammer.analyze import plot_ttvs

		nplanets = self.system.nplanets_fit
		measured = self.system.measured
		epoch = self.system.epoch
		error = self.system.error
		model, model_epoch, __ = model(self.theta_max, self.system)
		filename = self.output_path + "TTVs_" + self.date + '.png'
		plot_ttvs(nplanets, measured, epoch, error, model, model_epoch, filename)

	def autocorr(self):
		filename = self.output_path + "autocor_" + self.date + '.txt'
		# autocor  = open(self.output_path + "/autocor_"+self.date+".txt", "w+")
		try:
			tau = self.sampler.get_autocorr_time(discard=self.discard)
		except Exception as e:
			print(str(e))
			tau = e
		with open(filename, 'w') as f:
			f.write(str(tau))
		return tau

	def summarize(self):
		space = """

            """
		summary = """
            nwalkers: %i walkers
            niter_total: %i total iterations  
			nburnin: %i
            The resulting chain was thinned by a factor of %i

            The resultant orbital elements are below:

            """ % (self.nwalkers, self.niter+self.nburnin, self.nburnin, self.thin)
		run_description = open(self.output_path + "/run_description_" + self.date + ".txt", "w+")
		run_description.write(summary)
		run_description.write(space)
		for i in range(len(self.system.variable_labels)):
			run_description.write(str(self.system.variable_labels[i]) + ": " + str(self.theta_max[i]))
			run_description.write(space)
			# TODO: Verify this works
		# theta=self.theta_max
		# run_description.write(str(self.theta_max))
		print(summary)
		print(self.system.variable_labels)
		print(self.theta_max)

	def save_misc_txt(self, filename, info, more_info=None):
		filename = self.output_path + filename + "_" + self.date + '.txt'
		with open(filename, 'w') as f:
			f.write(str(info))
			if more_info != None:
				f.write(more_info)
		return
