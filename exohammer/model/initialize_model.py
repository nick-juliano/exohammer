#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:03:26 2021

@author: nickjuliano
"""

from exohammer.model.model_both import model_both as both
from exohammer.model.model_ttv import model_ttv as ttv
from exohammer.model.model_rv import model_rv as rv


def initialize_model(System):
    if System.rvbjd is None and System.epoch is not None:
        model = ttv
    elif System.rvbjd is not None and System.epoch is None:
        model = rv
    else:
        model = both

    return model
