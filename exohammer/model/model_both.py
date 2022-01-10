def model_both(theta, system):
    from exohammer.utilities import generate_planets 
    import ttvfast
    import numpy as np
    dt=0.4
    mstar            = system.mstar 
    epoch            = system.epoch
    measured         = system.measured
    error            = system.error
    orbital_elements = system.orbital_elements
    tmin             = system.tmin-dt
    tmax             = system.tmax+dt
    rvbjd            = system.rvbjd
    nplanets         = system.nplanets_fit
    planets          = generate_planets(theta, system)
    au_per_day       = 1731460
    
    mod=None
    epo=None
    rv_model=None
    
    model    = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)
    
    rv_model = model['rv']
    rv_model = np.array(rv_model)*au_per_day


    mod=[]
    epo=[]
    model_index, model_epoch, model_time, _, _,  = model['positions']
    trim        = min(np.where(np.array(model_time) == -2.))[0]
    model_index = np.array(model_index[:trim])
    model_epoch = np.array(model_epoch[:trim])
    model_time  = np.array(model_time[:trim])
    
    for i in range(nplanets):
        idx        = np.where(model_index == float(i))
        epoch_temp = np.copy(np.array(epoch[i]))
        epoch_temp = epoch_temp[epoch_temp  <= max(model_epoch[idx])]
        model_temp = model_time[idx][epoch_temp]
        mod.append(model_temp.tolist())
        epo.append(epoch_temp.tolist())
            
    return mod, epo, rv_model