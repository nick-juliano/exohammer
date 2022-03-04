from exohammer.utilities import generate_planets, au_per_day
import ttvfast
import numpy as np


def model_both(theta, system):
    dt = 0.3

    model = ttvfast.ttvfast(generate_planets(theta, system),
                            system.mstar,
                            system.tmin - dt,
                            dt,
                            system.tmax + dt,
                            rv_times=system.rvbjd)

    rv_model = np.array(model['rv']) * au_per_day

    mod = []
    epo = []
    model_index, model_epoch, model_time, _, _, = model['positions']
    trim = min(np.where(np.array(model_time) == -2.))[0]
    model_index = np.array(model_index[:trim])
    model_epoch = np.array(model_epoch[:trim])
    model_time = np.array(model_time[:trim])

    for i in range(system.nplanets_ttvs):
        idx = np.where(model_index == float(i))
        epoch_temp = np.copy(np.array(system.epoch[i]))
        epoch_temp = epoch_temp[epoch_temp <= max(model_epoch[idx])]
        model_temp = model_time[idx][epoch_temp]
        mod.append(model_temp.tolist())
        epo.append(epoch_temp.tolist())

    return mod, epo, rv_model
