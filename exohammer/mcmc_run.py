# -*- coding: utf-8 -*-
"""
MCMC Runner Module

Provides the `MCMCRun` class, which wraps `emcee` functionality for
fitting planetary orbital models to TTV and RV data using ensemble MCMC.

Dependencies:
    - numpy
    - emcee
    - corner
    - matplotlib
    - multiprocessing
    - exohammer.system.System
    - exohammer.store.StoreRun
    - exohammer.utilities
"""

import multiprocessing
from os import path, makedirs, getcwd
from datetime import datetime
from time import perf_counter
from numpy import argmax

import emcee
import corner
import matplotlib.pyplot as plt

from exohammer.system import System
from exohammer.store import StoreRun
from exohammer.utilities import sampler_to_theta_max, bic, plot_rvs, plot_ttvs


class MCMCRun:
    """
    Runs and manages an MCMC exploration of a planetary system model using TTV/RV data.

    Attributes:
        output_path (str): Directory to store results.
        date (str): Unique identifier based on datetime.
        system (System): Initialized System object with model/data context.
        lnprob (callable): Log-probability function used for sampling.
    """

    def __init__(self, planetary_system, data, lnprob=None):
        """
        Initialize a new MCMC run environment.

        Args:
            planetary_system (PlanetarySystem): System configuration.
            data (Data): TTV and/or RV observational data.
            lnprob (callable, optional): Custom probability function (overrides default).
        """
        self.date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_path = path.abspath(getcwd()) + "/Output/"
        run_dir = path.join(base_path, f'run_{self.date}')
        makedirs(run_dir, exist_ok=True)
        self.output_path = run_dir + "/"

        self.system = System(planetary_system, data)
        self.lnprob = lnprob if lnprob else self.system.lnprob

    def explore_iteratively(self, total_iterations, checkpoints,
                            burnin_factor=0.2, thinning_factor=0.001,
                            moves=emcee.moves.DEMove(live_dangerously=True),
                            nwalkers=None, verbose=True, tune=True, silent=False):
        """
        Runs the MCMC sampling procedure in defined checkpoints.

        Args:
            total_iterations (int): Total steps to run.
            checkpoints (int): Interval between intermediate storage.
            burnin_factor (float): Fraction of samples to discard as burn-in.
            thinning_factor (float): Fraction to thin the chain by.
            moves (emcee.moves): Sampling strategy.
            nwalkers (int): Number of walkers (default = 3×ndim, rounded even).
            verbose (bool): Show tqdm progress bars.
            tune (bool): Enable emcee’s tuning.
            silent (bool): Suppress plots during run.
        """
        ndim = self.system.ndim
        self.nwalkers = nwalkers or (ndim * 3 + ndim * 3 % 2)  # even number
        self.niter = int(checkpoints)
        self.total_iterations = total_iterations
        self.burnin_factor = burnin_factor
        self.thinning_factor = thinning_factor
        self.discard = int(self.niter * burnin_factor)
        self.thin = int(self.niter * thinning_factor)
        self.tune = tune
        self.silent = silent
        self.moves = moves
        self.pos = self.system.initial_state(self.nwalkers)

        num_cores = max(1, multiprocessing.cpu_count() // 2)

        with multiprocessing.get_context("fork").Pool(processes=num_cores) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                ndim,
                self.lnprob,
                args=[self.system],
                moves=self.moves,
                live_dangerously=True,
                pool=pool
            )

            nrepeat = total_iterations // checkpoints
            times = []

            for i in range(nrepeat):
                print(f"Run {i+1} of {nrepeat}, {checkpoints} steps")
                tic = perf_counter()
                self.sampler = sampler
                pos, prob, state = sampler.run_mcmc(
                    self.pos, checkpoints,
                    progress=verbose,
                    tune=self.tune,
                    skip_initial_state_check=True
                )
                toc = perf_counter()
                times.append(toc - tic)
                print(f"Steps completed: {(i+1)*checkpoints}, Time: {toc - tic:.2f}s")

                self.pos, self.prob, self.state = pos, prob, state
                sampler_to_theta_max(self)
                bic(self)

                store = StoreRun(self)
                store.store_csvs()

                self.plot_chains()
                self.autocorr()
                self.plot_ttvs()
                self.plot_rvs()
                self.summarize()
                self.plot_corner()

                self.niter += checkpoints
                self.discard = int(self.niter * burnin_factor)
                self.thin = int(self.niter * thinning_factor)

        print("Run complete")

    def plot_corner(self, samples=None, save=True):
        """
        Generate a corner plot of parameter posteriors.

        Args:
            samples (array, optional): Custom sample set.
            save (bool): Whether to save the plot.
        """
        samples = samples or self.sampler.get_chain(flat=True, discard=self.discard, thin=self.thin)
        fig = corner.corner(
            samples,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            labels=self.system.variable_labels,
            title_fmt=".2e"
        )
        if save:
            fig.savefig(self.output_path + f"corner_{self.date}.png")
        if not self.silent:
            plt.show()
        plt.close('all')

    def plot_chains(self, samples=None, save=True):
        """
        Plot the evolution of each parameter chain.

        Args:
            samples (array, optional): Custom MCMC chain samples.
            save (bool): Whether to save the figure.
        """
        samples = samples or self.sampler.get_chain(discard=self.discard, thin=self.thin)
        fig, axes = plt.subplots(len(self.system.variable_labels), figsize=(20, 30), sharex=True)
        fig.suptitle('Parameter Chains', fontsize=30)

        for i, label in enumerate(self.system.variable_labels):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, samples.shape[0])
            ax.set_ylabel(label)
            ax.yaxis.set_label_coords(-0.1, 0.5)

        if save:
            plt.savefig(self.output_path + f"Chains_{self.date}.png")
        if not self.silent:
            plt.show()
        plt.close('all')

    def plot_rvs(self):
        """
        Plot RV data and model fit.
        """
        _, _, rv_model = self.system.model(self.theta_max, self.system)
        filename = self.output_path + f"rvs_{self.date}.png"
        plot_rvs(self.system.rvbjd, self.system.rvmnvel, self.system.rverrvel, rv_model, filename, self.silent)

    def plot_ttvs(self):
        """
        Plot TTVs and model fit for each planet.
        """
        nplanets = self.system.nplanets_ttvs
        measured = self.system.measured
        epoch = self.system.epoch
        error = self.system.error
        mod, model_epoch, _ = self.system.model(self.theta_max, self.system)
        filename = self.output_path + f"TTVs_{self.date}.png"
        plot_ttvs(nplanets, measured, epoch, error, mod, model_epoch, filename, self.silent)

    def autocorr(self):
        """
        Compute and save the autocorrelation time estimate.

        Returns:
            autocorr_time or Exception: Autocorrelation time or error raised by emcee.
        """
        filename = self.output_path + f"autocor_{self.date}.txt"
        try:
            tau = self.sampler.get_autocorr_time(discard=self.discard)
        except Exception as e:
            tau = str(e)
            print("Autocorr error:", tau)
        with open(filename, 'w') as f:
            f.write(str(tau))
        return tau

    def summarize(self):
        """
        Save a summary text file describing the run configuration and results.
        """
        summary_path = self.output_path + f"run_description_{self.date}.txt"
        logprob_max_idx = argmax(self.sampler.get_log_prob(flat=True, thin=self.thin))

        header = f"""
nwalkers: {self.nwalkers}
niter_total: {self.niter}
nburnin: {self.discard}
thinning factor: {self.thin}
BIC: {self.bic}
max lnprob index: {logprob_max_idx}
Orbital elements:
"""

        with open(summary_path, "w") as f:
            f.write(header + "\n\n")
            for label, value in zip(self.system.variable_labels, self.theta_max):
                f.write(f"{label}: {value}\n")
