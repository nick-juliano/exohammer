#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:04:17 2021

@author: nickjuliano
"""

import os
import datetime
import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt

from exohammer.system import system
from exohammer.store import store_run
from exohammer.analyze import plot_rvs, plot_ttvs
from exohammer.utilities import model, sampler_to_theta_max, bic
import time
from exohammer.lnprob.initialize_prob import initialize_prob
from exohammer.model.initialize_model import initialize_model
import cProfile
import pstats

class mcmc_run:
	"""
	A class used to define the parameters of an MCMC run and explore the parameter space
	...
	Attributes
	----------
	output_path : str
		##
	date : str
		##
	EnsembleSampler : object
		##
	system : object
		##
	nwalkers : int
		##
	nburnin : int
		##
	thin : int
		##
	niter : int
		##
	moves : object
		##
	discard : int
        ##
    tune : bool
        ##
    sampler : object
        ##
    pos : array
        ##
    prob : array
        ##
    state : array
        ##
    samples : array
        ##
    theta_max : list
        ##
    Methods
    -------
    explore(iterations, thin=1, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)], verbose=True, tune=True)
        explores
    explore_again(niter, verbose=True)
        explores again
    plot_corner(samples=None, save=True)
        plots corner
    plot_chains(discard=None, samples=None, save=True)
        plots chains
    summarize()
        summarizes
    autocorr()
        autocorrelates
    plot_rvs()
        plots RVs
    plot_ttvs()
        plots TTVs
    """


	def __init__(self, planetary_system, data, lnprob=None):
		"""
        Parameters
        ----------
        planetary_system : object
            exohammer.planetary_system.planetary_system object
        data : object
            exohammer.data.data object
        lnprob : function
            probability function. Must be of the format lnprob(theta, args). Optional (determined by initialize_prob.py if None)
        """

		date = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time())[:-7]
		output_path = str(os.path.abspath(os.getcwd()) + "/Output/")
		testdir = output_path + 'run_' + date
		if not os.path.exists(testdir):
			os.makedirs(testdir)
		output_path = output_path + 'run_' + date + "/"
		self.output_path = output_path
		self.date = date
		self.system = system(planetary_system, data)
		if lnprob == None:
			self.lnprob = initialize_prob(system=self.system)

		self.system.model = initialize_model(system=self.system)
	def explore(self, iterations, thin=1, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
	            verbose=True, tune=True, silent=False):

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

		walk = self.system.ndim * 3
		self.nwalkers = int(walk) if ((int(walk) % 2) == 0) else int(walk) + 1
		self.nburnin = int(0.1 * iterations)
		self.thin = thin
		self.niter = int(iterations - self.nburnin)
		self.moves = moves
		self.discard = 0
		self.tune = tune
		self.silent=silent

		filename = self.output_path + "tutorial.h5"
		self.backend = emcee.backends.HDFBackend(filename)
		self.backend.reset(self.nwalkers, self.system.ndim)

		p0 = self.system.initial_state(self.nwalkers)
		# Initialize the sampler
		sampler = emcee.EnsembleSampler(self.nwalkers,
		                               self.system.ndim,
		                               self.system.lnprob,
		                               args=[self.system],
		                               moves=self.moves,
		                                backend=self.backend)  # , backend=backend)

		print("Running burn-in... " + str(self.nburnin) + " steps")

		p0, _, _ = sampler.run_mcmc(p0,
		                            self.nburnin,
		                            progress=verbose)

		sampler.reset()

		print("Running production..." + str(self.niter) + " steps")
		pos, prob, state = sampler.run_mcmc(p0,
		                                    self.niter,
		                                    thin=self.thin,
		                                    progress=verbose,
		                                    tune=tune)

		self.sampler = sampler
		self.pos = pos
		self.prob = prob
		self.state = state

		#sampler_to_theta_max(self)
		#bic(self)

	def explore_again(self, niter, verbose=True):
		print("Running production..." + str(niter) + " steps")
		self.niter += niter
		pos, prob, state = self.sampler.run_mcmc(self.pos, niter, thin=self.thin, progress=verbose, tune=self.tune)
		self.pos = pos  # np.append(self.pos, pos.reshape(len(pos),len(pos[-1])), axis=0)
		self.prob = prob
		self.state = state

		sampler_to_theta_max(self)
		bic(self)

	def explore_iteratively(self, total_iterations,
	                        checkpoints,
	                        burnin_factor=.2,
	                        thinning_factor=.001,
	                        moves=emcee.moves.DEMove(),
		                    verbose=True, tune=True, silent=False):

		walk = self.system.ndim * 3
		self.nwalkers = int(walk) if ((int(walk) % 2) == 0) else int(walk) + 1
		self.niter = int(checkpoints)
		self.moves = moves
		self.discard = int(self.niter * burnin_factor)
		self.thin = int(self.niter * thinning_factor)
		self.tune = tune
		self.silent = silent
		self.total_iterations=total_iterations
		self.pos = self.system.initial_state(self.nwalkers)
		# Initialize the sampler
		#backend = emcee.backends.HDFBackend(self.output_path + "chain.h5")
		#backend.reset(self.nwalkers, self.system.ndim)

		sampler = emcee.EnsembleSampler(self.nwalkers,
		                               self.system.ndim,
		                               self.lnprob,
		                               args=[self.system],
		                               moves=moves)  # , backend=backend)
		nrepeat=int(total_iterations/checkpoints)

		completed = 0
		times=[]
		# prof = cProfile.Profile()
		for i in range(nrepeat):
			# prof.enable()
			tic = time.perf_counter()
			print("Steps completed: " + str(completed))
			print("Run " + str(i) + " of " + str(nrepeat) + ", " +str(checkpoints)+ " steps")
			self.sampler=sampler
			pos, prob, state = sampler.run_mcmc(self.pos, checkpoints, progress=verbose, tune=self.tune)
			toc = time.perf_counter()
			times.append(toc-tic)
			print(times)
			sampler_to_theta_max(self)
			bic(self)
			self.pos=pos
			self.prob=prob
			self.state=state
			run=self
			store=store_run(run)
			store.serialize()
			run.plot_chains()
			run.autocorr()
			run.plot_ttvs()
			run.plot_rvs()
			run.summarize()
			run.plot_corner()

			sampler_to_theta_max(self)
			self.niter += int(checkpoints)
			self.discard = int(self.niter * burnin_factor)
			self.thin = int(self.niter * thinning_factor)
			completed+=checkpoints
			# prof.disable()
			# stats = pstats.Stats(prof).sort_stats('tottime')
			# stats.print_stats(.1)
		print("Run complete")

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

		filename = self.output_path + "corner_" + self.date + '.png'
		if samples == None:
			samples = self.sampler.get_chain(flat=True, discard=self.discard, thin=self.thin)
		figure = corner.corner(samples, labels=self.system.variable_labels)
		if save == True:
			figure.savefig(filename)
		if self.silent != True:
			plt.show()
		plt.close('all')

	def plot_chains(self, samples=None, save=True):
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
		filename = self.output_path + "Chains_" + self.date + '.png'
		# samples = self.samples
		if samples == None:
			samples = self.sampler.get_chain(discard=self.discard, thin=self.thin)
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
		if self.silent != True:
			plt.show()
		plt.close('all')

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

	def plot_rvs(self):
		__, __, rv_model = model(self.theta_max, self.system)
		filename = self.output_path + "rvs_" + self.date + '.png'
		plot_rvs(self.system.rvbjd, self.system.rvmnvel, self.system.rverrvel, rv_model, filename, self.silent)

	def plot_ttvs(self):
		nplanets = self.system.nplanets_ttvs
		measured = self.system.measured
		epoch = self.system.epoch
		error = self.system.error
		mod, model_epoch, __ = model(self.theta_max, self.system)
		filename = self.output_path + "TTVs_" + self.date + '.png'
		plot_ttvs(nplanets, measured, epoch, error, mod, model_epoch, filename, self.silent)

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
			The bic is: %i
            The resultant orbital elements are below:

            """ % (self.nwalkers, self.niter, self.discard, self.thin, self.bic)

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
