#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:37:11 2021

@author: nickjuliano
"""

class store_run:
    def __init__(self, pkl_object):
        # self.moves = pkl_object.moves
        # self.discard = pkl_object.discard
        # self.sampler = pkl_object.sampler
        # self.pos = pkl_object.pos
        # self.prob = pkl_object.prob
        # self.state = pkl_object.state
        # self.samples = pkl_object.samples
        # self.theta_max = pkl_object.theta_max
        # self.output_path = pkl_object.output_path
        # self.date = pkl_object.date
        # self.nplanets = pkl_object.system.nplanets
        # self.orbital_elements = pkl_object.system.orbital_elements
        # #self.x = self.given, self.planetary_system
        # self.EnsembleSampler = pkl_object.EnsembleSampler
        # self.system = pkl_object.system
        self.run = pkl_object
        # def generate(self):
        #     import numpy as np
        #     """
        #     Generates planets for ttvfast

        #     Parameters
        #     ----------
        #         theta : array
        #             The variables to fit within the MCMC
        #         x     : array
        #             All other arguments of the following form:
        #                 mstar : float            # x[0]
        #                 epoch : array            # x[1]
        #                 measured : array         # x[2]
        #                 error : array            # x[3]
        #                 orbital_elements : array # x[4]

        #     Returns
        #     -------
        #         planets : list
        #             A planets instance for ttvfast in the form [planet1, planet2, ..., planeti]
        #     """
        #     orbital_elements = self.run.system.orbital_elements
        #     p0              = []
        #     fixed           = []
        #     theta_min       = []
        #     theta_max       = []
        #     variable_labels = []
        #     fixed_labels    = []
        #     gauss           = []
        #     sigma           = []
        #     mu              = []
        #     index           = []
        #     non_gaus        = []
        #     non_gaus_max    = []
        #     non_gaus_min    = []
        #     theta_ranges    = []
        #     options=[]

        #     planet_designation = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        #     k=0
        #     for i in orbital_elements:
        #         element=orbital_elements[i]
        #         if len(element) == 1:
        #             fixed_labels.append(str(i))
        #             k+=1
        #             fixed.append(element[0])
        #             non_gaus.append(element[0])
        #             non_gaus_max.append(element[0]+(element[0]/(1.e9)))
        #             non_gaus_min.append(element[0]-(element[0]/(1.e9)))

                    
        #         if len(element) == 2:
        #             minimum=element[0]
        #             maximum=element[1]
        #             options=np.linspace(minimum,  maximum, 200)
        #             theta_ranges.append(options)
        #             theta_min.append(minimum)
        #             theta_max.append(maximum)
        #             p0.append(np.random.choice(options))
        #             variable_labels.append(str(i))
        #             non_gaus.append(element)
        #             non_gaus_max.append(maximum)
        #             non_gaus_min.append(minimum)
                    
        #         if len(element) == 3:
        #             index.append((i-k)+(len(orbital_elements)))
        #             mu.append(element[0])
        #             sigma.append(element[1])
                    
        #     self.p0              = p0
        #     self.ndim            = len(p0)
        #     self.fixed           = fixed
        #     self.min_theta       = theta_min
        #     self.max_theta       = theta_max
        #     self.variable_labels = variable_labels
        #     self.fixed_labels    = fixed_labels
        #     self.sigma           = sigma
        #     self.mu              = mu
        #     self.index           = index
        #     self.non_gaus        = non_gaus
        #     self.non_gaus_max    = non_gaus_max
        #     self.non_gaus_min    = non_gaus_min
        #     self.theta_ranges    = theta_ranges
        # # generate(self)
        
        # system=pkl_object.system
        # self.mstar      = system.mstar

        # self.epoch      = system.epoch
        # self.measured   = system.measured
        # self.error      = system.error
        # self.rvbjd      = system.rvbjd
        # self.rvmnvel    = system.rvmnvel
        # self.rverrvel   = system.rverrvel
        # # self.nplanets_fit = len(self.epoch)
        # self.nplanets = len(self.orbital_elements)/7
        # self.rvbjd=system.rvbjd
        # self.rvmnvel=system.rvmnvel
        # self.rverrvel=system.rverrvel
        # print(self.epoch)
        # def find_minmax(self):
        #     if self.epoch!=None: 
        #         x=self.measured
                
        #         tmin=min(x[0])
        #         tmax=max(x[0])
                
        #         for i in x:
        #             if min(i)<tmin:
        #                 tmin=min(i)
        #             if max(i)>tmax:
        #                 tmax=max(i)
                        
        #         if self.rvbjd!=None:
        #             if min(self.rvbjd) < tmin:
        #                 tmin = min(self.rvbjd)
        #             if max(self.rvbjd) > tmax:
        #                 tmax = max(self.rvbjd)   
        #         self.tmin=tmin
        #         self.tmax=tmax+200
                
        #     elif self.epoch==None:   
        #         if self.rvbjd!=None:
        #             x=self.rvbjd
        #             self.tmin=min(x)
        #             self.tmax=max(x)
        
        # def best_fit(X, Y):
        #     xbar = sum(X)/len(X)
        #     ybar = sum(Y)/len(Y)
        #     n = len(X) # or len(Y)

        #     numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
        #     denum = sum([xi**2 for xi in X]) - n * xbar**2

        #     m = numer / denum
        #     b = ybar - m * xbar

        #     return m, b
                
        # def ttvs(measured, epoch):
        #     import numpy as np
        #     oc=[]
        #     for i in range(len(measured)):
        #         slope, inter = best_fit(epoch[i], measured[i])
        #         y = inter + slope * np.array(epoch[i])
        #         oc.append((measured[i]-y))
        #     self.computed_ttvs = oc
            
        # find_minmax(self)
        # if self.measured!=None:
        #     ttvs(self.measured, self.epoch)
        # else:
        #     self.computed_ttvs=None
        
    def store(self):
        import pickle
        pkl_object = self
        filename= self.run.output_path + "pickle_" + self.run.date +'.obj'
        file = open(filename, 'wb') 
        pickle.dump(pkl_object, file)
        
        

def restore(filename=None):
    import pickle 
    import easygui
    if filename==None:
        #import easygui
        filename=easygui.fileopenbox()
        file = open(filename, 'rb') 
        run = pickle.load(file)
    else:
        file = open(filename, 'rb') 
        run = pickle.load(file)
    return run