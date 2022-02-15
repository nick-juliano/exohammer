#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:02:01 2021

@author: nickjuliano
"""

class planetary_system:
    def __init__(self, nplanets_ttvs, nplanets_rvs, orbital_elements, theta=None):
        
        self.orbital_elements = orbital_elements
        self.nplanets_rvs = nplanets_rvs
        self.nplanets_ttvs = nplanets_ttvs

        if type(theta) == list:
            self.theta = theta
        def generate(self):
            import numpy as np
            """
            Generates planets for ttvfast

            Parameters
            ----------
                theta : array
                    The variables to fit within the MCMC
                x     : array
                    All other arguments of the following form:
                        mstar : float            # x[0]
                        epoch : array            # x[1]
                        measured : array         # x[2]
                        error : array            # x[3]
                        orbital_elements : array # x[4]

            Returns
            -------
                planets : list
                    A planets instance for ttvfast in the form [planet1, planet2, ..., planeti]
            """
            orbital_elements = self.orbital_elements
            p0              = []
            fixed           = []
            theta_min       = []
            theta_max       = []
            variable_labels = []
            fixed_labels    = []
            gauss           = []
            sigma           = []
            mu              = []
            index           = []
            non_gaus        = []
            non_gaus_max    = []
            non_gaus_min    = []
            theta_ranges    = []
            options=[]

            planet_designation = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            k=0
            self.fixed_flag = False
            self.flat_flag = False
            self.gaus_flag = False

            for i in orbital_elements:
                element=orbital_elements[i]
                if len(element) == 1:
                    self.fixed_flag = True
                    fixed_labels.append(str(i))
                    k+=1
                    fixed.append(element[0])
                    non_gaus.append(element[0])
                    non_gaus_max.append(element[0]+(element[0]/(1.e9)))
                    non_gaus_min.append(element[0]-(element[0]/(1.e9)))

                    
                if len(element) == 2:
                    self.flat_flag = True
                    minimum=element[0]
                    maximum=element[1]
                    options=np.linspace(minimum,  maximum, 200)
                    theta_ranges.append(options)
                    theta_min.append(minimum)
                    theta_max.append(maximum)
                    p0.append(np.random.choice(options))
                    variable_labels.append(str(i))
                    non_gaus.append(element)
                    non_gaus_max.append(maximum)
                    non_gaus_min.append(minimum)
                    
                if len(element) == 3:
                    self.gaus_flag == True
                    index.append((i-k)+(len(orbital_elements)))
                    mu.append(element[0])
                    sigma.append(element[1])
                    
            self.p0              = p0
            self.ndim            = len(p0)
            self.fixed           = fixed
            self.theta_min       = theta_min
            self.theta_max       = theta_max
            self.variable_labels = variable_labels
            self.fixed_labels    = fixed_labels
            self.sigma           = sigma
            self.mu              = mu
            self.index           = index
            self.non_gaus        = non_gaus
            self.non_gaus_max    = non_gaus_max
            self.non_gaus_min    = non_gaus_min
            self.theta_ranges    = theta_ranges
        generate(self)
        
    
    def describe(self):
        print('nplanets: ', self.nplanets)
        print()
        print('nplanets_fit: ', self.nplanets_fit)
        print()
        print('orbital_elements: ', self.orbital_elements)
        print()
        print('p0: ', self.p0)
        print()
        print('ndim: ', self.ndim)
        print()
        print('fixed: ', self.fixed)
        print()
        print('theta_min: ', self.theta_min)
        print()
        print('theta_max: ', self.theta_max)
        print()
        print('variable_labels: ', self.variable_labels)
        print()
        print('fixed_labels: ', self.fixed_labels)
        print()
        print('sigma: ', self.sigma)
        print()
        print('mu: ', self.mu)
        print()
        print('index: ', self.index)
        print()
        print('non_gaus: ', self.non_gaus)
        print()
        print('non_gaus_max: ', self.non_gaus_max)
        print()
        print('non_gaus_min: ', self.non_gaus_min)