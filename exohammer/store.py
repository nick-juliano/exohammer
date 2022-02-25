#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:37:11 2021

@author: nickjuliano
"""
import pickle
import easygui

class store_run:
    def __init__(self, pkl_object):

        self.run = pkl_object
        
    def serialize(self):
        import pickle
        pkl_object = self
        filename= self.run.output_path + "pickle_" + self.run.date +'.obj'
        file = open(filename, 'wb') 
        pickle.dump(pkl_object, file)

def restore(filename=None):
    if filename==None:
        filename=easygui.fileopenbox()
        file = open(filename, 'rb') 
        run = pickle.load(file)
    else:
        file = open(filename, 'rb') 
        run = pickle.load(file)
    return run