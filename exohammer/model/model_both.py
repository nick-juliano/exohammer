# -*- coding: utf-8 -*-

from ttvfast import ttvfast
from numpy import array, where, copy, where
from exohammer.utilities import generate_planets, au_per_day


def model_both(theta, system):
    dt = 0.3
    model = ttvfast(generate_planets(theta, system),
                            system.m_star,
                            system.tmin - dt,
                            dt,
                            system.tmax + dt,
                            rv_times=system.rvbjd)

    rv_model = array(model['rv']) * au_per_day

    mod = []
    epo = []
    model_index, model_epoch, model_time, _, _, = model['positions']
    sentinel_indices = where(model_time == -2.)[0]
    trim = sentinel_indices[0] if len(sentinel_indices) > 0 else len(model_time)
    model_index = array(model_index[:trim])
    model_epoch = array(model_epoch[:trim])
    model_time = array(model_time[:trim])

    for i in range(system.nplanets_ttvs):
        idx = where(model_index == float(i))
        epoch_temp = copy(array(system.epoch[i]))
        epoch_temp = epoch_temp[epoch_temp <= max(model_epoch[idx])]
        model_temp = model_time[idx][epoch_temp]
        mod.append(model_temp.tolist())
        epo.append(epoch_temp.tolist())

    return mod, epo, rv_model
