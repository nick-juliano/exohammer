#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:03:26 2021

@author: nickjuliano
"""

import numpy as np
from exohammer.utilities import *
#import ttvfast 
import imp
import ttvfast
import numpy as np

def initialize_model(system):
    from exohammer.model.model_both import model_both as both
    from exohammer.model.model_ttv import model_ttv as ttv
    from exohammer.model.model_rv import model_rv as rv
    
    if system.rvbjd == None and system.epoch != None:
        model = ttv
    elif system.rvbjd != None and system.epoch == None:
        model = rv
    elif system.rvbjd != None and system.epoch != None:
        model = both
    
    return model