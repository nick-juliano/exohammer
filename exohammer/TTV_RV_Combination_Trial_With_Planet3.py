def best_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    if denum ==0:
        print("this error occured")
        print()
        print('X (epoch): ', X)
        print('Y (measured): ', Y)
        print()
        print('xbar (avg epoch): ', xbar)
        print('ybar (avg measured): ', ybar)
        print('n (epoch length): ', n)
        print()
        print('numer: ', numer)
        print('denum:', denum)
    m = numer / denum
    b = ybar - m * xbar
    return m, b  



def compute_oc(obs, epoch):
    from numpy import polyfit
    slope, inter = polyfit(epoch,obs,1)
    tlin = inter + slope * np.array(epoch)
    oc = obs - tlin
    return oc




def ttvs(measured, epoch):
    import numpy as np
    oc=[]
    for i in range(len(measured)):
        slope, inter = best_fit(epoch[i], measured[i])
        y = inter + slope * np.array(epoch[i])
        oc.append((measured[i]-y))
    return oc





def trim(x, model, epo, flatten=True):
    data, planetary_system = x
    epoch    = data.epoch
    measured = data.measured
    error    = data.error
    ttvs     = data.computed_ttvs
    nplanets = planetary_system.nplanets
    mod  = []
    meas = []
    err  = []
    ep   = []
    tt   = []
    for i in range(nplanets):
        if len(measured[i])>len(model[i]):
            meas.append(measured[i][:len(epo[i])])
            tt.append(ttvs[i][:len(epo[i])])
            mod.append(model[i])
            err.append(error[i][:len(epo[i])])
            ep.append(epo[i])
        if len(model[i])>len(measured[i]):
            meas.append(measured[i])
            tt.append(ttvs[i])
            mod.append(model[i][:len(epoch[i])])
            err.append(error[i])
            ep.append(epoch[i])
        if len(model[i])==len(measured[i]):
            meas.append(measured[i])
            tt.append(ttvs[i])
            mod.append(model[i])
            err.append(error[i])
            ep.append(epoch[i])
    if flatten==True:
        mod  = np.array(flatten_list(mod))
        meas = np.array(flatten_list(meas))
        err  = np.array(flatten_list(err))
        ep   = np.array(flatten_list(ep))
    return mod, meas, tt, err, ep




def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list   


import ttvfast
from ttvfast import models

def generate_planets(theta, x):
    # import ttvfast
    # from ttvfast import models
    data, planetary_system = x
    mstar            = data.mstar 
    epoch            = data.epoch
    measured         = data.epoch
    error            = data.error
    orbital_elements = data.orbital_elements
    tmin             = data.tmin
    tmax             = data.tmax
    nplanets         = planetary_system.nplanets
    index            = planetary_system.index
    minimum          = planetary_system.theta_min
    maximum          = planetary_system.theta_max
    mu               = planetary_system.mu
    sigma            = planetary_system.sigma
    fixed_labels = planetary_system.fixed_labels
    fixed_values = planetary_system.fixed
    variable_labels = planetary_system.variable_labels
    orb_elements = []
    for i in range(len(fixed_values)):
        orb_elements.append({'element' : fixed_labels[i],
                                'value' : fixed_values[i]})
    for i in range(len(variable_labels)):
        orb_elements.append({'element' : variable_labels[i],
                                'value' : theta[i]})
    planets = []
    planet_designation = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    for j in range(nplanets):
        for i in orb_elements:
            if i['element'] == 'mass_' + planet_designation[j]:
                mass         = i['value']
            if i['element'] == 'period_' + planet_designation[j]:
                period       = i['value']
            if i['element'] == 'eccentricity_' + planet_designation[j]:
                eccentricity = i['value'] 
            if i['element'] == 'inclination_' + planet_designation[j]:
                inclination  = i['value'] 
            if i['element'] == 'longnode_' + planet_designation[j]:
                longnode     = i['value'] 
            if i['element'] == 'argument_' + planet_designation[j]:
                argument     = i['value'] 
            if i['element'] == 'mean_anomaly_' + planet_designation[j]:
                mean_anomaly = i['value'] 
        planet=models.Planet(
                        mass=mass,                       # M_sun
                        period=period,                   # days
                        eccentricity=eccentricity,
                        inclination=inclination,         # degrees
                        longnode=longnode,               # degrees
                        argument=argument,               # degrees
                        mean_anomaly=mean_anomaly,       # degrees
                        )
        planets.append(planet)
    return planets
        
        
        
        
def model_transits(theta, x):
    # import ttvfast
    # from ttvfast import models
    import numpy as np
    data, planetary_system = x
    mstar            = data.mstar 
    epoch            = data.epoch
    measured         = data.epoch
    error            = data.error
    orbital_elements = data.orbital_elements
    tmin             = data.tmin
    tmax             = data.tmax
    nplanets         = planetary_system.nplanets
    index            = planetary_system.index
    minimum          = planetary_system.theta_min
    maximum          = planetary_system.theta_max
    mu               = planetary_system.mu
    sigma            = planetary_system.sigma
    planets          = generate_planets(theta, x)
    dt=0.01
    model_index, model_epoch, model_time, _, _,  = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax)['positions']
    trim        = min(np.where(np.array(model_time) == -2.))[0]
    model_index = np.array(model_index[:trim])
    model_epoch = np.array(model_epoch[:trim])
    model_time  = np.array(model_time[:trim])
    mod=[]
    epo=[]
    for i in range(nplanets):
        idx        = np.where(model_index == float(i))
        epoch_temp = np.copy(np.array(epoch[i]))
        epoch_temp = epoch_temp[epoch_temp  <= max(model_epoch[idx])]
        model_temp = model_time[idx][epoch_temp]
        mod.append(model_temp.tolist())
        epo.append(epoch_temp.tolist())
    return mod, epo

def model_rvs(theta, x):
    # import ttvfast
    # from ttvfast import models
    import numpy as np
    data, planetary_system = x
    mstar            = data.mstar 
    epoch            = data.epoch
    measured         = data.epoch
    error            = data.error
    orbital_elements = data.orbital_elements
    tmin             = data.tmin
    tmax             = data.tmax
    nplanets         = planetary_system.nplanets
    index            = planetary_system.index
    minimum          = planetary_system.theta_min
    maximum          = planetary_system.theta_max
    mu               = planetary_system.mu
    sigma            = planetary_system.sigma
    planets          = generate_planets(theta, x)
    rvbjd            = data.rvbjd
    dt=0.01
    model  = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)['rv']
    model  = np.array(model)*au_per_day
    return model

class given:
    def __init__(self, mstar, 
                 ttvs,              #[epoch, measured, error, given_ttvs], 
                 rvs,               #[bjd, mnvel, errvel], 
                 orbital_elements):
                 
        self.mstar      = mstar
        
        self.epoch      = ttvs[0]
        self.measured   = ttvs[1]
        self.error      = ttvs[2]        
        self.given_ttvs = ttvs[3]
        
        self.rvbjd      = rvs[0]
        self.rvmnvel    = rvs[1]
        self.rverrvel   = rvs[2]
        
        self.orbital_elements = orbital_elements
        self.nplanets_fit = len(self.epoch)

        def find_minmax(self):
            x=self.measured
            
            tmin=min(x[0])
            for i in x:
                if min(i)<tmin:
                    tmin=min(i)
            
            if min(self.rvbjd) < tmin:
                tmin = min(self.rvbjd) 

            tmax=max(x[0])
            for i in x:
                if max(i)>tmax:
                    tmax=max(i)
            if max(self.rvbjd) > tmax:
                tmax = max(self.rvbjd)   
            
            self.tmin=tmin
            self.tmax=tmax+200
        
        def best_fit(X, Y):
            xbar = sum(X)/len(X)
            ybar = sum(Y)/len(Y)
            n = len(X) # or len(Y)

            numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
            denum = sum([xi**2 for xi in X]) - n * xbar**2

            m = numer / denum
            b = ybar - m * xbar

            return m, b
                
        def ttvs(measured, epoch):
            oc=[]
            for i in range(len(measured)):
                slope, inter = best_fit(epoch[i], measured[i])
                y = inter + slope * np.array(epoch[i])
                oc.append((measured[i]-y))
            self.computed_ttvs = oc
            
        find_minmax(self)
        ttvs(measured, epoch)




import numpy as np

class planetary_system:
    def __init__(self, nplanets, orbital_elements, theta=None):
        self.orbital_elements = orbital_elements
        self.nplanets = nplanets
        def generate(self):
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
            for i in orbital_elements:
                element=orbital_elements[i]
                if len(element) == 1:
                    fixed_labels.append(str(i))
                    k+=1
                    fixed.append(element[0])
                    non_gaus.append(element[0])
                    non_gaus_max.append(element[0]+(element[0]/(1.e9)))
                    non_gaus_min.append(element[0]-(element[0]/(1.e9)))

                    
                if len(element) == 2:
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
            self.options         = options
        generate(self)
        
    def initial_state(self, nwalkers):
        initial_state=[]
        orbital_elements=self.orbital_elements
        for i in range(nwalkers):
            p0=[]
            for j in orbital_elements:
                element=orbital_elements[j]
                if len(element) == 2:
                    minimum=element[0]
                    maximum=element[1]
                    options=np.linspace(minimum,  maximum, 200)
                    p0.append(np.random.choice(options))
            initial_state.append(p0)
        return initial_state
    
    def describe(self):
        print('nplanets: ', self.nplanets)
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


    
from astropy import constants as const

mearth = const.M_earth.cgs.value #grams
msun   = const.M_sun.cgs.value
au_per_day     = 1731460 #meters per second

mstar  = 1.034 
orbital_elements = {'mass_b'         : [mearth/msun,  10 * mearth/msun ],   
                    'period_b'       : [13.849-.05,   13.849+.05       ],                
                    'eccentricity_b' : [0.00,         0.04             ],
                    'inclination_b'  : [89.5,         90.5             ],
                    'longnode_b'     : [-10.0,        10.0             ],
                    'argument_b'     : [0.0,          360.0            ],
                    'mean_anomaly_b' : [0.0,          360.0            ],
     
                    'mass_c'         : [mearth/msun,  10 * mearth/msun ],   
                    'period_c'       : [16.2369-.05,  16.2369+.05      ],                
                    'eccentricity_c' : [0.00,         0.04             ],
                    'inclination_c'  : [89.5,         90.5             ],
                    'longnode_c'     : [0.0                            ],
                    'argument_c'     : [0.0,          360.0            ],
                    'mean_anomaly_c' : [0.0,          360.0            ]}
kepler_36=planetary_system(2, orbital_elements, theta=None)



class prob_functions:
    def __init__(self):
        return
    
    
    def compute_oc(obs, epoch):
        from numpy import polyfit
        slope, inter = polyfit(epoch,obs,1)
        tlin = inter + slope * np.array(epoch)
        oc = obs - tlin
        return oc
        
    def lnprob(theta, x):
        def model_ttvs(theta, x):
            import ttvfast
            from ttvfast import models
            def generate_planets(theta, x):
                data, planetary_system = x
                mstar            = data.mstar 
                epoch            = data.epoch
                measured         = data.epoch
                error            = data.error
                orbital_elements = data.orbital_elements
                tmin             = data.tmin
                tmax             = data.tmax
                nplanets         = planetary_system.nplanets
                index            = planetary_system.index
                minimum          = planetary_system.theta_min
                maximum          = planetary_system.theta_max
                mu               = planetary_system.mu
                sigma            = planetary_system.sigma
                fixed_labels = planetary_system.fixed_labels
                fixed_values = planetary_system.fixed
                variable_labels = planetary_system.variable_labels

                orb_elements = []
                for i in range(len(fixed_values)):
                    orb_elements.append({'element' : fixed_labels[i],
                                            'value' : fixed_values[i]})
                for i in range(len(variable_labels)):
                    orb_elements.append({'element' : variable_labels[i],
                                            'value' : theta[i]})
                planets = []

                planet_designation = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

                for j in range(nplanets):
                    for i in orb_elements:
                        if i['element'] == 'mass_' + planet_designation[j]:
                            mass         = i['value']
                        if i['element'] == 'period_' + planet_designation[j]:
                            period       = i['value']
                        if i['element'] == 'eccentricity_' + planet_designation[j]:
                            eccentricity = i['value'] 
                        if i['element'] == 'inclination_' + planet_designation[j]:
                            inclination  = i['value'] 
                        if i['element'] == 'longnode_' + planet_designation[j]:
                            longnode     = i['value'] 
                        if i['element'] == 'argument_' + planet_designation[j]:
                            argument     = i['value'] 
                        if i['element'] == 'mean_anomaly_' + planet_designation[j]:
                            mean_anomaly = i['value'] 
                    planet=models.Planet(
                                    mass=mass,                       # M_sun
                                    period=period,                   # days
                                    eccentricity=eccentricity,
                                    inclination=inclination,         # degrees
                                    longnode=longnode,               # degrees
                                    argument=argument,               # degrees
                                    mean_anomaly=mean_anomaly,       # degrees
                                    )
                    planets.append(planet)
                return planets


            data, planetary_system = x
            mstar            = data.mstar 
            epoch            = data.epoch
            measured         = data.epoch
            error            = data.error
            orbital_elements = data.orbital_elements
            tmin             = data.tmin
            tmax             = data.tmax
            nplanets_fit     = data.nplanets_fit
            
            rvbjd            = data.rvbjd
            rvmnvel          = data.rvmnvel 
            rverrvel         = data.rverrvel 
            
            nplanets         = planetary_system.nplanets
            index            = planetary_system.index
            minimum          = planetary_system.theta_min
            maximum          = planetary_system.theta_max
            mu               = planetary_system.mu
            sigma            = planetary_system.sigma

            planets=generate_planets(theta, x)
            #print(planets)

            dt=0.25

            mod=[]
            epo=[]

            if rvbjd==None:
                model_index, model_epoch, model_time, _, _,  = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax)['positions']
                trim        = min(np.where(np.array(model_time) == -2.))[0]
                model_index = np.array(model_index[:trim])
                model_epoch = np.array(model_epoch[:trim])
                model_time  = np.array(model_time[:trim])
                mod=[]
                epo=[]
                for i in range(nplanets_fit):
                    idx        = np.where(model_index == float(i))
                    epoch_temp = np.copy(np.array(epoch[i]))
                        #print(theta)
                    trimming   = epoch_temp  <= max(model_epoch[idx])
                    epoch_temp = epoch_temp[trimming]

                    model_temp = model_time[idx][epoch_temp]

                    mod.append(model_temp.tolist())
                    epo.append(epoch_temp.tolist())

                rv_model=None
                
            if rvbjd!=None:
                ttvfastresults=ttvfast.ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)
                #print(ttvfastresults)
                model_index, model_epoch, model_time, _, _,  = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax)['positions']
                #print(model_index)
                trim        = min(np.where(np.array(model_time) == -2.))[0]
                model_index = np.array(model_index[:trim])
                #print(model_index)
                model_epoch = np.array(model_epoch[:trim])
                model_time  = np.array(model_time[:trim])
                mod=[]
                epo=[]
                for i in range(nplanets_fit):
                    idx        = np.where(model_index == float(i))
                    epoch_temp = np.copy(np.array(epoch[i]))
                        #print(theta)
                    if len(model_epoch[idx])!=0:
                        
                        trimming   = epoch_temp  <= max(model_epoch[idx])
                        epoch_temp = epoch_temp[trimming]
    
                        model_temp = model_time[idx][epoch_temp]
    
                        mod.append(model_temp.tolist())
                        epo.append(epoch_temp.tolist())

                rv_model=ttvfastresults['rv']

            return mod, epo, rv_model
        
        
        def lnprior(theta, x):
            flat = theta.copy().flatten()
            data, planetary_system = x
            index            = planetary_system.index
            mstar            = data.mstar 
            minimum          = planetary_system.theta_min
            maximum          = planetary_system.theta_max
            mu               = planetary_system.mu
            sigma            = planetary_system.sigma
            
            for i in range(len(flat),0,-1):
                for j in index:
                    if i==j:
                        flat=np.delete(flat, j)

            lp = 0. if np.all(minimum < flat) and  np.all(flat < maximum) else -np.inf

            gaus = theta[index]
            for i in range(len(index)):
                g = (((gaus[i] - mu[i] ) / sigma[i] )**2.)*-.5
                lp +=  g
            return lp
        
        def lnlike(theta, x):
            
            def best_fit(X, Y):
                xbar = sum(X)/len(X)
                ybar = sum(Y)/len(Y)
                n = len(X) # or len(Y)

                numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
                denum = sum([xi**2 for xi in X]) - n * xbar**2
                if denum ==0:
                    print("this error occured")
                    print()
                    print('X (epoch): ', X)
                    print('Y (measured): ', Y)
                    print()
                    print('xbar (avg epoch): ', xbar)
                    print('ybar (avg measured): ', ybar)
                    print('n (epoch length): ', n)
                    print()
                    print('numer: ', numer)
                    print('denum:', denum)
                m = numer / denum
                b = ybar - m * xbar

                return m, b
            
            def ttvs(measured, epoch):
                oc=[]
                for i in range(len(measured)):
                    slope, inter = best_fit(epoch[i], measured[i])
                    y = inter + slope * np.array(epoch[i])
                    oc.append((measured[i]-y))
                return oc

            def trim(x, model, epo, flatten=True):
                data, planetary_system = x
                    
                epoch    = data.epoch
                measured = data.measured
                error    = data.error
                ttvs     = data.computed_ttvs
                nplanets = planetary_system.nplanets
                nplanets_fit = data.nplanets_fit
                    
                mod  = []
                meas = []
                err  = []
                ep   = []
                tt   = []
                for i in range(nplanets_fit):
                    if len(measured[i])>len(model[i]):
                        meas.append(measured[i][:len(epo[i])])
                        tt.append(ttvs[i][:len(epo[i])])
                        mod.append(model[i])
                        err.append(error[i][:len(epo[i])])
                        ep.append(epo[i])

                    if len(model[i])>len(measured[i]):
                        meas.append(measured[i])
                        tt.append(ttvs[i])
                        mod.append(model[i][:len(epoch[i])])
                        err.append(error[i])
                        ep.append(epoch[i])
                    if len(model[i])==len(measured[i]):
                        meas.append(measured[i])
                        tt.append(ttvs[i])
                        mod.append(model[i])
                        err.append(error[i])
                        ep.append(epoch[i])

                if flatten==True:
                    mod  = np.array(flatten_list(mod))
                    meas = np.array(flatten_list(meas))
                    err  = np.array(flatten_list(err))
                    ep   = np.array(flatten_list(ep))
                    
                return mod, meas, tt, err, ep
            
            def flatten_list(_2d_list):
                flat_list = []
                # Iterate through the outer list
                for element in _2d_list:
                    if type(element) is list:
                        # If the element is of type list, iterate through the sublist
                        for item in element:
                            flat_list.append(item)
                    else:
                        flat_list.append(element)
                return flat_list   
            
            model, epo, rv_model = model_ttvs(theta, x)

            mod, meas, tt, err, ep = trim(x, model, epo, flatten=False)
            
            obs=[]
            comp=[]
            data, planetary_system = x
            nplanets         = planetary_system.nplanets
            nplanets_fit = data.nplanets_fit 
            for i in range(nplanets_fit):
                #obs.append(meas[i])
                #comp.append(ep[i])
                obs.append(meas[i])
                comp.append(mod[i])
            resid          = np.array(flatten_list(obs))-np.array(flatten_list(comp))    
            likelihood     = (np.array(resid)**2.)/(np.array(flatten_list(err))**2.)

            sum_likelihood=0
            
            for i in likelihood:
                sum_likelihood += i
            
            if rv_model!=None:
                #print('rv', rv_model)
                rvresid=np.array(flatten_list(data.rvmnvel))-(np.array(flatten_list(rv_model))*au_per_day)
                rv_likelihood=(np.array(rvresid)**2.)/(np.array(flatten_list(data.rverrvel))**2.)
            
                for i in rv_likelihood:
                    sum_likelihood+=i
                    
            likelihood = -0.5 * sum_likelihood

            if not np.isfinite(likelihood):
                likelihood = -np.inf    
            return likelihood
        
        lp = lnprior(theta, x)


        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + lnlike(theta, x)      
    
                
func=prob_functions




class mcmc_run:
    import emcee
    import os
    def __init__(self, planetary_system, given, prob_functions):
        import os
        import datetime
        import emcee
        date        = str(datetime.datetime.now().date())+'_'+str(datetime.datetime.now().time())[:-7]
        output_path = str(os.path.abspath(os.getcwd()) + "/Output/")
        os.mkdir(output_path + 'run_' + date)
        output_path=output_path+'run_'+ date+"/"
        self.output_path = output_path
        self.date        = date
        self.planetary_system=planetary_system
        self.given=given
        self.prob_functions = prob_functions
        x = given, planetary_system
        self.x = x
        self.EnsembleSampler = emcee.EnsembleSampler

    # def __init__(self, nwalkers, niter, discard=0, thin=1, nburnin=None, moves = emcee.moves.DEMove()):
    #     import emcee
    #     import os
    #     import datetime
        
    #     date        = str(datetime.datetime.now().date())+'_'+str(datetime.datetime.now().time())[:-7]
    #     output_path = str(os.path.abspath(os.getcwd()) + "/Output/")
        
    #     os.mkdir(output_path + 'run_' + date)
        
    #     output_path=output_path+'run_'+ date+"/"
        
    #     self.output_path = output_path
    #     self.date        = date
    #     self.nwalkers    = nwalkers
    #     self.niter       = niter
    #     self.thin        = thin
    #     self.discard     = discard
    #     self.moves       = moves
    #     if nburnin==None:
    #         self.nburnin = 0.2*niter
    #     else:
    #         self.nburnin = nburnin
    def explore(self, niter, thin_by=1, moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)], verbose=True, tune=True):
        #import emcee
        walk=self.planetary_system.ndim*3
        self.nwalkers       = int(walk) if ((int(walk) % 2) == 0) else int(walk) + 1
        self.nburnin        = int(0.2*niter)
        self.thin_by        = thin_by
        if thin_by>1:
            self.niter      = niter/thin_by
        else:
            self.niter      = niter
            
        #self.thin           = thin
        self.moves          = moves
        self.discard        = 0
        prob_functions      = self.prob_functions
        x                   = self.x
        ndim                = self.planetary_system.ndim
        nwalkers            = self.nwalkers
        nburnin             = self.nburnin
        thin_by             = self.thin_by
        p0=self.planetary_system.initial_state(nwalkers)
        # Initialize the sampler
        sampler = self.EnsembleSampler(nwalkers, ndim, prob_functions.lnprob, args=[x], moves=moves)#, backend=backend)
        # Run the Burn In
        #print("Running burn-in...")
        #p0, _, _ = sampler.run_mcmc(p0, nburnin, progress=verbose)
        ## ^^^VVV COMMENT THIS OUT TO APPLY BURNIN TO SAMPLES ^^^
        #sampler.reset()
        # Run the Production
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, thin_by=thin_by, progress=verbose, tune=tune)
        self.sampler   = sampler 
        self.pos       = pos 
        self.prob      = prob 
        self.state     = state 
        
        def sampler_to_theta_max(self):
             sampler=self.sampler
             #samples = sampler.get_chain(flat=True, thin=self.thin, discard=self.discard)
             #samples = sampler.get_chain(flat=True, thin=self.thin, discard=self.nburnin)
             samples = sampler.get_chain(flat=True, discard=self.nburnin)
             self.samples = samples
             #old VVV
             # self.theta_max = samples[np.argmax(sampler.flatlnprobability)]
             #new VVV
             self.theta_max = samples[np.argmax(sampler.get_log_prob(discard=run.nburnin, flat=True))]
        sampler_to_theta_max(self) 
        
    # def explore(self, planetary_system, given, prob_functions=prob_functions, verbose=True):
    #     import emcee
           
    #     self.planetary_system=planetary_system
    #     self.given=given
    #     x = given, planetary_system
    #     self.x = x
    #     self.prob_functions=prob_functions
    #     nwalkers = self.nwalkers
    #     ndim=planetary_system.ndim
    #     niter = self.niter
    #     nburnin = self.nburnin
    #     thin = self.thin
    #     p0=planetary_system.initial_state(nwalkers)
        
    #     # Initialize the sampler
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_functions.lnprob, args=[x], moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])#, backend=backend)

    #     print("Running burn-in...")
    #     p0, _, _ = sampler.run_mcmc(p0, nburnin, progress=verbose)
    #     sampler.reset()

    #     print("Running production...")
    #     pos, prob, state = sampler.run_mcmc(p0, niter, progress=verbose, thin=thin, tune=True)
    #     self.sampler   = sampler 
    #     self.pos       = pos 
    #     self.prob      = prob 
    #     self.state     = state 
    #     def sampler_to_theta_max(sampler):
    #         samples = sampler.flatchain
    #         samples[np.argmax(sampler.flatlnprobability)]
    #         theta_max=samples[np.argmax(sampler.flatlnprobability)]
    #         return theta_max
    #     self.theta_max = sampler_to_theta_max(sampler)
    # def __init__(self, planetary_system, given, prob_functions):
    #     import os
    #     import datetime
    #     import emcee
    #     date        = str(datetime.datetime.now().date())+'_'+str(datetime.datetime.now().time())[:-7]
    #     output_path = str(os.path.abspath(os.getcwd()) + "/Output/")
    #     os.mkdir(output_path + 'run_' + date)
    #     output_path=output_path+'run_'+ date+"/"
    #     self.output_path = output_path
    #     self.date        = date
    #     self.planetary_system=planetary_system
    #     self.given=given
    #     self.prob_functions = prob_functions
    #     x = given, planetary_system
    #     self.x = x
    #     self.EnsembleSampler = emcee.EnsembleSampler
        
    # def __init__(self, nwalkers, nburnin, niter, thin=0, discard=0, moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]):
    #     # import emcee
    #     # import os
    #     # import datetime
        
    #     date        = str(datetime.datetime.now().date())+'_'+str(datetime.datetime.now().time())[:-7]
    #     output_path = str(os.path.abspath(os.getcwd()) + "/Output/")
        
    #     os.mkdir(output_path + 'run_' + date)
        
    #     output_path=output_path+'run_'+ date+"/"
        
    #     self.output_path = output_path
    #     self.date        = date
    #     self.nwalkers    = nwalkers
    #     #self.nburnin     = nburnin
    #     self.niter       = niter
    #     self.thin        = thin
    #     self.discard     = nburnin
    #     self.moves       = moves
        
    # def explore(self, nwalkers, nburnin, niter, thin, moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)], verbose=True, tune=True):
    #     from . import prob_functions
    #     import ttvfast
    #     from ttvfast import models
    #     import numpy as np
    #     #import emcee
    #     self.nwalkers       = nwalkers
    #     self.nburnin        = nburnin
    #     self.niter          = niter
    #     self.thin           = thin
    #     self.moves          = moves        
    #     prob_functions      = self.prob_functions
    #     x                   = self.x
    #     ndim                = self.planetary_system.ndim
    #     p0=self.planetary_system.initial_state(nwalkers)
    #     # Initialize the sampler
    #     sampler = self.EnsembleSampler(nwalkers, ndim, prob_functions.lnprob, args=[x], moves=moves)#, backend=backend)
    #     # Run the Burn In
    #     print("Running burn-in...")
    #     p0, _, _ = sampler.run_mcmc(p0, nburnin, progress=verbose)
    #     sampler.reset()
    #     # Run the Production
    #     print("Running production...")
    #     pos, prob, state = sampler.run_mcmc(p0, niter, progress=verbose, tune=tune)
    #     self.sampler   = sampler 
    #     self.pos       = pos 
    #     self.prob      = prob 
    #     self.state     = state 
    #     self.theta_max = sampler_to_theta_max(sampler)
    
    # def explore(self, planetary_system, given, prob_functions, verbose=True):
    #     import emcee
           
    #     self.planetary_system=planetary_system
    #     self.given=given
    #     x = given, planetary_system
    #     self.x = x
    #     self.prob_functions=prob_functions
    #     moves = self.moves
    #     nwalkers = self.nwalkers
    #     ndim=planetary_system.ndim
    #     niter = self.niter
    #     #nburnin = self.nburnin
    #     thin = self.thin
    #     p0=planetary_system.initial_state(nwalkers)
        
    #     # Initialize the sampler
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_functions.lnprob, args=[x], moves=moves)#, backend=backend)
        
    #     # print("Running burn-in...")
    #     # p0, _, _ = sampler.run_mcmc(p0, self.discard, progress=verbose)
    #     # sampler.reset()

    #     print("Running production...")
    #     pos, prob, state = sampler.run_mcmc(p0, niter/thin, thin_by=thin, progress=verbose, tune=True)
    #     self.sampler   = sampler 
    #     self.pos       = pos 
    #     self.prob      = prob 
    #     self.state     = state 
    #     def sampler_to_theta_max(self):
    #         sampler=self.sampler
    #         samples = sampler.get_chain(flat=True, thin=self.thin, discard=self.discard)
    #         self.samples = samples
    #         self.theta_max = samples[np.argmax(sampler.flatlnprobability)]
    #     sampler_to_theta_max(self)
    
    def plot_ttvs(self, theta=None):
        """
        Parameters
        ----------
            theta : array
                The fitted (or to be fit) variables within the MCMC
                (here, it may be wise to use theta = theta_max = samples[np.argmax(sampler.flatlnprobability)])

            x     : array
                All other arguments of the following form:
                    mstar : float
                    epoch : array
                    measured : array
                    error : array
                    orbital_elements :array

        Returns
        -------
        Plots of each planet's ttvs
        """
        import matplotlib.pyplot as plt
        #from .utilities import model_transits, ttvs, compute_oc
        x = self.x
        data=self.given
        given, planetary_system = x
        mstar            = given.mstar 
        epoch            = given.epoch
        measured         = given.measured
        error            = given.error
        orbital_elements = given.orbital_elements
        tmin             = given.tmin
        tmax             = given.tmax
        given_ttvs       = given.given_ttvs
        nplanets         = planetary_system.nplanets
        nplanets         = planetary_system.nplanets
        output_path      = self.output_path
        date             = self.date
        theta            = self.theta_max
        mod, epo = model_transits(theta, x)
        planet_designation = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        period=[theta[1], theta[8]]                # [period_b, period_c]
        filename= self.output_path + "ttvs_" + self.date + '.png'
        fig, axes = plt.subplots(nplanets, figsize=(20, 16), sharex=True)
        fig.suptitle('ttvs', fontsize=30)
        given_ttv = data.given_ttvs
        ttv_model = ttvs(mod, epo)
        ttv_data  = ttvs(measured, epoch)
        for i in range(nplanets):
            oc_data      = ttv_data[i]
            oc_model     = ttv_model[i]
            planet       = planet_designation[i]
            ax = axes[i]
            ax.set_title('O-C Kepler 36-'+planet+ ' with Fitted Orbital Elements', fontsize=20) #OPTIONAL: \n Burn In = 1,000, Iterations = 10,000', fontsize=20)
            ax.errorbar(epoch[i], oc_data, error[i], fmt ='o', label='TTV From Measured Transits')
            ax.errorbar(epoch[i], given_ttv[i], error[i], fmt ='o', label='TTV From Holczer et. al.')
            ax.plot(epo[i], oc_model, label='Best Fit (O-C)')
            ax.legend(fontsize=16)
        fig.savefig(filename)
        plt.show()
        
    def plot_logprob(self):
        import matplotlib.pyplot as plt
        filename= self.output_path + "logprob_" + self.date +'.png'
        logprob=self.sampler.get_log_prob(flat=True, discard=self.discard)
        fig, axes = plt.subplots(len(self.planetary_system.variable_labels), figsize=(20, 30), sharex=True)
        fig.suptitle('logprob', fontsize=30)
        for i in range(len(self.planetary_system.variable_labels)):
            try:
                ax = axes[i]
                ax.plot(logprob[:, :, i ], "k", alpha=0.3)
                ax.set_xlim(0, len(logprob))
                ax.set_ylabel(self.planetary_system.variable_labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            except:
                break
        plt.savefig(filename)
        plt.show()
        
    def plot_corner(self):
        import corner
        filename= self.output_path + "corner_" + self.date + '.png'
        samples = self.samples
        figure = corner.corner(samples, labels=self.planetary_system.variable_labels)
        figure.savefig(filename)



    def plot_chains(self):
        import matplotlib.pyplot as plt
        filename= self.output_path + "Chains_" + self.date +'.png'
        #samples = self.samples
        samples = self.sampler.get_chain(discard=self.discard)
        fig, axes = plt.subplots(len(self.planetary_system.variable_labels), figsize=(20, 30), sharex=True)
        fig.suptitle('chains', fontsize=30)
        for i in range(len(self.planetary_system.variable_labels)):
            ax = axes[i]
            ax.plot(samples[:, :, i ], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.planetary_system.variable_labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        plt.savefig(filename)
        plt.show()
        
        
        
    def summarize(self):
        space = """

            """
        summary="""
            This run sampled %i walkers with %i iterations with a burn-in of %i.  

            The chain was thinned by %i

            The resultant orbital elements are below:

            """%(self.nwalkers, self.niter, self.discard, self.thin_by)
        run_description = open(self.output_path + "/run_description_"+self.date+".txt", "w+")
        run_description.write(summary)
        run_description.write(space)
        run_description.write(str(self.planetary_system.variable_labels))
        run_description.write(space)
        run_description.write(str(self.theta_max))
        print(summary)
        print(self.planetary_system.variable_labels)
        print(self.theta_max)
        
        
        
    def autocorr(self):
        filename = self.output_path + "autocor_" + self.date + '.txt'
        #autocor  = open(self.output_path + "/autocor_"+self.date+".txt", "w+")
        try:
            tau        = self.sampler.get_autocorr_time(discard=self.discard)
        except Exception as e:
            print(str(e))
            tau = e
        with open(filename, 'w') as f:
            f.write(str(tau))
        return tau
    
    
    
    def pickle(self):
        import pickle
        pkl_object = self
        filename= self.output_path + "pickle_" + self.date +'.obj'
        file = open(filename, 'wb') 
        pickle.dump(pkl_object, file)
        
    def plot_rvs(self, theta=None):
        """
        Parameters
        ----------
            theta : array
                The fitted (or to be fit) variables within the MCMC
                (here, it may be wise to use theta = theta_max = samples[np.argmax(sampler.flatlnprobability)])

            x     : array
                All other arguments of the following form:
                    mstar : float
                    epoch : array
                    measured : array
                    error : array
                    orbital_elements :array

        Returns
        -------
        Plots of each planet's rvs
        """
        import matplotlib.pyplot as plt
        #from .utilities import model_transits, ttvs, compute_oc
        x = self.x
        data=self.given
        given, planetary_system = x
        mstar            = given.mstar 
        epoch            = given.epoch
        measured         = given.measured
        error            = given.error
        orbital_elements = given.orbital_elements
        tmin             = given.tmin
        tmax             = given.tmax
        given_ttvs       = given.given_ttvs
        nplanets         = planetary_system.nplanets
        nplanets         = planetary_system.nplanets
        
        output_path      = self.output_path
        date             = self.date
        theta            = self.theta_max
        rvbjd            = given.rvbjd
        rvmnvel          = given.rvmnvel 
        rverrvel         = given.rverrvel 
        rv_model              = model_rvs(theta, x)
        filename= self.output_path + "rvs_" + self.date + '.png'
        #plt.figure(figsize=(20, 16))
        plt.errorbar(rvbjd, rvmnvel, rverrvel, fmt ='o', label='RVs From Keck Data')
        plt.plot(rvbjd, rv_model, 'or', label='best fit')
        plt.legend(fontsize=16)
        plt.title("RVs", fontsize=20)
        plt.savefig(filename)
        plt.show()
        
def sampler_to_theta_max(sampler):
    #import numpy as np
    samples = sampler.flatchain
    samples[np.argmax(sampler.flatlnprobability)]
    theta_max=samples[np.argmax(sampler.flatlnprobability)]
    return theta_max

def restore(mcmc_run, filename=None):
    import pickle 
    if filename==None:
        #import easygui
        filename=easygui.fileopenbox()
        file = open(filename, 'rb') 
        run = pickle.load(file)
    else:
        file = open(filename, 'rb') 
        run = pickle.load(file)



from Input_Measurements import *

from Input_Measurements import rv

kepler_36 = planetary_system(2, orbital_elements, theta=None)
data      = given(mstar, [epoch, measured, error, ttv_holczer], rv, orbital_elements)

run       = mcmc_run(kepler_36, data, prob_functions)
# import ttvfast

# mstar            = data.mstar 
# epoch            = data.epoch
# measured         = data.epoch
# error            = data.error
# orbital_elements = data.orbital_elements
# tmin             = data.tmin
# tmax             = data.tmax
# tmax=2458723.006589
# rvbjd            = data.rvbjd
# rvmnvel          = data.rvmnvel 
# rverrvel         = data.rverrvel 

# nplanets         = kepler_36.nplanets
# index            = kepler_36.index
# minimum          = kepler_36.theta_min
# maximum          = kepler_36.theta_max
# mu               = kepler_36.mu
# sigma            = kepler_36.sigma
# theta = kepler_36.p0

# x=[data, kepler_36]
# planets=generate_planets(theta, x)

# dt=0.25

# mod=[]
# epo=[]
# ttvfastresults=ttvfast.ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)

run.explore(niter=5000)

run.pickle()
run.summarize()
run.autocorr()
run.plot_chains()
run.plot_corner()
run.plot_ttvs()
run.plot_rvs()






