#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:01:22 2021

@author: nickjuliano
"""

class data:
    def __init__(self, mstar, 
                 ttvs,              #[epoch, measured, error], 
                 rvs,               #[bjd, mnvel, errvel], 
                 orbital_elements):
        
        import numpy as np
        self.mstar      = mstar
        
        self.epoch      = ttvs[0]
        self.measured   = ttvs[1]
        self.error      = ttvs[2]        
        
        self.rvbjd      = rvs[0]
        self.rvmnvel    = rvs[1]
        self.rverrvel   = rvs[2]
        

        
        self.orbital_elements = orbital_elements
        # self.nplanets_fit = len(self.epoch)

        def find_minmax(self):
            if self.epoch!=None: 
                x=self.measured
                
                tmin=min(x[0])
                tmax=max(x[0])
                
                for i in x:
                    if min(i)<tmin:
                        tmin=min(i)
                    if max(i)>tmax:
                        tmax=max(i)
                        
                if self.rvbjd!=None:
                    if min(self.rvbjd) < tmin:
                        tmin = min(self.rvbjd)
                    if max(self.rvbjd) > tmax:
                        tmax = max(self.rvbjd)   
                self.tmin=tmin
                self.tmax=tmax+200
                
            elif self.epoch==None:   
                if self.rvbjd!=None:
                    x=self.rvbjd
                    self.tmin=min(x)
                    self.tmax=max(x)

            
        find_minmax(self)
        #self..tmin=2454955.917878-0.1  
        # ttvs(self.measured, self.epoch)