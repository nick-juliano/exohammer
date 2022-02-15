from exohammer.utilities import *
import numpy as np

def lnprob(theta, system):
    
    ########
    ########
    
    ########
    ########
    def lnprior(theta, system):

        flat = theta.copy().flatten()
        index            = system.index
        minimum          = system.theta_min
        maximum          = system.theta_max
        mu               = system.mu
        sigma            = system.sigma
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
    
    def lnlike(theta, system):

        ttmodel, epo, rv_model = system.model(theta,system)
        sum_likelihood=0
        ttv_likelihood=0
        rv_likelihood=0
        
        mod, meas, err, ep = trim(system.nplanets_ttvs, system.epoch, system.measured, ttmodel, system.error, flatten=False)
        obs=[]
        comp=[]
        nplanets_rvs     = system.nplanets_rvs
        nplanets_ttvs = system.nplanets_ttvs
        for i in range(nplanets_ttvs):
            obs.append(meas[i])
            comp.append(mod[i])
        obs = flatten_list(obs)
        comp = flatten_list(comp)
        err = flatten_list(err)

        resid = np.array(obs) - np.array(comp)

        if len(resid) == len(err):
            ttv_likelihood = (np.array(resid) ** 2.) / (np.array(err) ** 2.)

        else:
            ttv_likelihood = [-np.inf]

        for i in ttv_likelihood:
            sum_likelihood += i

        likelihood = -0.5 * sum_likelihood

        if not np.isfinite(likelihood):
            likelihood = -np.inf
        return likelihood
            
            
        return likelihood
    
    lp = lnprior(theta, system)
    llike=lnlike(theta, system)
    
    if not np.isfinite(lp) and not np.isfinite(llike):
        return -np.inf
    else:
        return lp + llike