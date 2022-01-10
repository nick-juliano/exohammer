def model_rv(theta, system):
    from exohammer.utilities import generate_planets
    import ttvfast
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

            
    return mod, epo, rv_model