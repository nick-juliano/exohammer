#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:57:17 2021

@author: nickjuliano
"""
from . import utilities
from . import data
from . import planetary_system
from . import mcmc_run
from . import system
from . import store

def describe():
	print(""""
            TTV_Module
            ________________________________________________________
            ____GIVEN____
            ________________________________________________________
           
            TTV_Module.given.given: class
            organizes the given information for the rest of the classes and functions
            .__init__(self, mstar, 
                      ttvs,              #[epoch, measured, error, given_ttvs], 
                      rvs,               #[bjd, mnvel, errvel], 
                      orbital_elements)
            
            RETURNS:
                #### planetary system data:
                self.mstar
                self.orbital_elements
                self.nplanets_fit (total number of planets to fit TTVS)
                self.nplanets (total number of planets in the system)
                self.tmin
                self.tmax
                
                #### ttv data:
                self.epoch
                self.measured
                self.error
                self.given_ttvs
                self.computed_ttvs
                
                #### rv data:
                self.rvbjd
                self.rvmnvel
                self.rverrvel
                self.orbital_elements
                
            ________________________________________________________
            ____Input_Measurements____
            ________________________________________________________
            TTV_Module.Input_Measurements: module
            
            .mearth:
                mass of the earth
            .msun:
                mass of the sun
            .mstar:
                mass of the star
            .measured:
                measured TTV mid-points (list, bjd) -- [[planet_b transits], [planet_c transits], etc]
            .epoch:
                measured TTV mid-point epochs (list, intergers) -- [[planet_b epochs], [planet_c epochs], etc]
                note: included in case of missing transit mid-points
            .error:
                measured TTV mid-point errors (list, days) -- [[planet_b errors], [planet_c errors], etc]
            .ttv_holczer:
                Holczer et. al. calculated TTVs mid-point epochs (list, days) -- [[planet_b ttvs], [planet_c ttvs], etc]
                note: included in case of missing transit mid-points
            .orbital_elements
                (dictionary) of each orbital element with _i denoting associated planet including the following
                mass, period, eccentricity, inclination, longnode, argument, mean_anomaly
                note: all units in degrees, solar mass, and days
                note: the value of each element must be [], single value inside denotes fixed, two denote min and max
            .x:
                (list) [mstar, epoch, measured, error, orbital_elements]
            .rv:
                (list) [[rvbjd], [mnvel], [errvel]]
                
            ________________________________________________________
            ____planetary_system____
            ________________________________________________________
            
            TTV_Module.planetary_system.planetary_system: class
            
            FUNCTIONS:
                .__init__(self, nplanets, orbital_elements, theta=None)
                    RETURNS:
                        self.orbital_elements
                        self.nplanets
                        self.nplanets_fit
                        self.p0
                        self.ndim
                        self.fixed
                        self.theta_min
                        self.theta_max
                        self.variable_labels
                        self.fixed_labels
                        self.sigma
                        self.mu
                        self.index
                        self.non_gaus
                        self.non_gaus_max
                        self.non_gaus_min
                        self.theta_ranges
                    
                .initial_state(self, nwalkers)
                    RETURNS:
                        initial_state i.e. p0
                
                .describe(self):
                    Prints a description of the planetary system
            
            ________________________________________________________
            ____prob_functions____
            ________________________________________________________
            
            TTV_Module.prob_functions.prob_functions: class
            
            FUNCTIONS:
                .compute_oc(obs, epoch):
                    computes the O-C
                    
                .lnprob(theta, x)
                    RETURNS:
                        lnprior+lnlike
                
                    .lnprior(theta, x):
                        RETURNS lnprior
                        
                    .lnlike(theta, x):
                        RETURNS likelihood
            
            ________________________________________________________
            ____mcmc_run____
            ________________________________________________________
            
            TTV_Module.prob_functions.prob_functions: class
            
            FUNCTIONS:
                .__init__(self, planetary_system, given, prob_functions):
                    RETURNS:
                        self.output_path
                        self.date
                        self.planetary_system
                        self.given
                        self.prob_functions
                        self.x = given, planetary_system
                        self.EnsembleSampler
                    
                .explore(self, niter, thin_by=1, moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)], verbose=True, tune=True)
                    PARAMETERS:
                        niter: iterations
                        thin_by: record every thin_by steps (thin_by*niter samples made)
                        moves: types of moves made in emcee
                        verbose: boolean of verbose computation
                        tune: adjusts guesses as it learns
                        
                    RETURNS:
                        self.moves
                        self.discard
                        self.sampler
                        self.pos
                        self.prob
                        self.state
                        self.samples
                        self.theta_max
                
                .plot_ttvs(self, theta=None):
                    PARAMETERS:
                        theta, optional, will plot TTVS given any orbital elements
                    RETURNS:
                        plot of each planet's TTVs
                        
                .plot_corner(self, theta=None)
                    RETURNS:
                        corner plot of the run
                        
                .plot_chains(self):
                    RETURNS:
                        plot of chains
                        
                .summarize(self):
                    RETURNS:
                        summary of run, results, and other relevant info. Also saves to file
                
                .autocorr(self):
                    RETURNS:
                        estimated Tau
                        
                pickle(self):
                    RETURNS:
                        saves pkl object to output_path
                        
                plot_rvs(self, theta=None):
                    RETURNS:
                        RV plot
                        
            ________________________________________________________
            ____utilities____
            ________________________________________________________
            
            TTV_Module.utilities: module
            
            .mearth
            .msun
            .au_per_day [meters/second]
            .mstar
            
            FUNCTIONS:
                .best_fit(X, Y):
                    RETURNS:
                        slope, intercept
                    
                .compute_oc(obs, epoch):
                    RETURNS:
                        Observed-Computed
                
                .ttvs(measured, epoch):
                    RETURNS:
                        computed ttvs
                        
                .trim(x, model, epo, flatten=True):
                    RETURNS:
                        mod, meas, tt, err, ep (all trimmed to length and epoch)
                        
                .flatten_list(_2d_list):
                    RETURNS:
                        flat_list
                        
                .generate_planets(theta, x):
                    RETURNS:
                        list of planet objects for ttvfast
                
                .model_transits(theta, x):
                    RETURNS:
                        [modelled transits], [epochs]
                        
                .model_rvs(theta, x):
                    RETURNS:
                        RVs [m/s] on given BJDs
                        
                .plot_rvs(self, theta=None):
                    RETURNS:
                        RV plot
                        
                .sampler_to_theta_max(sampler):
                    RETURNS:
                        theta_max
                        
                .restore(mcmc_run, filename=None)
                    RETURNS:
                        pickled run
                
                .plot_periodogram(rv, title):
                    RETURNS:
                        periodogram
                  """)
