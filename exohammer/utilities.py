#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:58:30 2021

@author: nickjuliano
"""

from astropy import constants as const
from numpy import polyfit
import numpy as np
from ttvfast import models
import ttvfast
import pickle
import matplotlib.pyplot as plt

def sun_to_earth(solar_mass):
    mearth     = const.M_earth.cgs.value #grams
    msun       = const.M_sun.cgs.value
    earth_mass = solar_mass/msun*mearth
    return earth_mass




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
    slope, inter = polyfit(epoch,obs,1)
    tlin = inter + slope * np.array(epoch)
    oc = obs - tlin
    return oc




def ttvs(measured, epoch):
    oc=[]
    for i in range(len(measured)):
        slope, inter = best_fit(epoch[i], measured[i])
        y = inter + slope * np.array(epoch[i])
        oc.append((measured[i]-y))
    return oc





def trim(nplanets, epoch, measured, model, error, flatten=True):
    mod  = []
    meas = []
    err  = []
    ep   = []
    for i in range(nplanets):
        if len(measured[i])>len(model[i]):
            meas.append(measured[i][:len(model[i])])
            mod.append(model[i])
            err.append(error[i][:len(epoch[i])])
            ep.append(epoch[i])
        elif len(model[i])>len(measured[i]):
            meas.append(measured[i])
            mod.append(model[i][:len(epoch[i])])
            err.append(error[i])
            ep.append(epoch[i])
        elif len(model[i])==len(measured[i]):
            meas.append(measured[i])
            mod.append(model[i])
            err.append(error[i])
            ep.append(epoch[i])
    if flatten==True:
        mod  = np.array(flatten_list(mod))
        meas = np.array(flatten_list(meas))
        err  = np.array(flatten_list(err))
        ep   = np.array(flatten_list(ep))
    return mod, meas, err, ep





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


def generate_planets_from_scratch(theta, planetary_system):

    nplanets         = planetary_system.nplanets
    fixed_labels     = planetary_system.fixed_labels
    fixed_values     = planetary_system.fixed
    variable_labels  = planetary_system.variable_labels
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
            if i['element'] == 'mass_'         + planet_designation[j]:
                mass         = i['value']
            if i['element'] == 'period_'       + planet_designation[j]:
                period       = i['value']
            if i['element'] == 'eccentricity_' + planet_designation[j]:
                eccentricity = i['value'] 
            if i['element'] == 'inclination_'  + planet_designation[j]:
                inclination  = i['value'] 
            if i['element'] == 'longnode_'     + planet_designation[j]:
                longnode     = i['value'] 
            if i['element'] == 'argument_'     + planet_designation[j]:
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
    #print(planets)
    return planets


def generate_planets(theta, system):


    nplanets         = system.nplanets_rvs
    fixed_labels     = system.fixed_labels
    fixed_values     = system.fixed
    variable_labels  = system.variable_labels
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
            if i['element'] == 'mass_'         + planet_designation[j]:
                mass         = i['value']
            elif i['element'] == 'period_'       + planet_designation[j]:
                period       = i['value']
            elif i['element'] == 'eccentricity_' + planet_designation[j]:
                eccentricity = i['value'] 
            elif i['element'] == 'inclination_'  + planet_designation[j]:
                inclination  = i['value'] 
            elif i['element'] == 'longnode_'     + planet_designation[j]:
                longnode     = i['value'] 
            elif i['element'] == 'argument_'     + planet_designation[j]:
                argument     = i['value'] 
            elif i['element'] == 'mean_anomaly_' + planet_designation[j]:
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
    #print(planets)
    return planets




def model(theta, system):
        dt=0.4
        mstar            = system.mstar 
        epoch            = system.epoch
        measured         = system.measured
        error            = system.error
        orbital_elements = system.orbital_elements
        tmin             = system.tmin-dt
        tmax             = system.tmax+dt
        rvbjd            = system.rvbjd
        nplanets_ttvs         = system.nplanets_ttvs
        planets          = generate_planets(theta, system)
        au_per_day       = 1731460
        
        mod=None
        epo=None
        rv_model=None
        
        model    = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)
        
        if rvbjd!=None:
            rv_model = model['rv']
            rv_model = np.array(rv_model)*au_per_day

        if epoch !=None:
            mod=[]
            epo=[]
            model_index, model_epoch, model_time, _, _,  = model['positions']
            trim        = min(np.where(np.array(model_time) == -2.))[0]
            model_index = np.array(model_index[:trim])
            model_epoch = np.array(model_epoch[:trim])
            model_time  = np.array(model_time[:trim])
            for i in range(nplanets_ttvs):
                idx        = np.where(model_index == float(i))
                epoch_temp = np.copy(np.array(epoch[i]))
                epoch_temp = epoch_temp[epoch_temp  <= max(model_epoch[idx])]
                model_temp = model_time[idx][epoch_temp]
                mod.append(model_temp.tolist())
                epo.append(epoch_temp.tolist())
                
        return mod, epo, rv_model
    
    
    

def model_rvs(theta, planetary_system, tmin, tmax, dt, rvbjd):
    mstar            = planetary_system.mstar
    planets          = generate_planets_from_scratch(theta, planetary_system)
    model  = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)['rv']
    rv_model  = np.array(model)*au_per_day
    return rv_model




def model_ttvs(theta, planetary_system, tmin, tmax):
    dt=0.3
    mstar            = planetary_system.mstar 
    tmin             = min(rvbjd)
    tmax             = max(rvbjd)
    planets          = generate_planets_from_scratch(theta, planetary_system)
    model  = ttvfast.ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)
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
    return mod, epo




def sampler_to_theta_max(sampler):
    samples = sampler.flatchain
    samples[np.argmax(sampler.flatlnprobability)]
    theta_max=samples[np.argmax(sampler.flatlnprobability)]
    return theta_max




def restore(mcmc_run, filename=None):
    if filename==None:
        #import easygui
        filename=easygui.fileopenbox()
        file = open(filename, 'rb') 
        run = pickle.load(file)
    else:
        file = open(filename, 'rb') 
        run = pickle.load(file)
    return run



def compute_rv_resid(rv_measured, rv_simulated):
    resid=rv_measured-rv_simulated
    return resid


def plot_periodogram(t, rv, title):
    t=np.array(t)
    t=t-t[0]
    rv=np.array(rv)
    s1=len(t)
    sx=0.0
    sx2=0.0
    sx=sum(rv)/len(rv)
    for m in range(s1):
        sx2=(rv[m]-sx)**2+sx2
    
    sx2=sx2/s1
    px=0
    p1=0
    p2=0
    p3=0
    p4=0
    p=0
    t1=0
    t2=0
    pstart=60
    pstop=t[-1]
    pint=1
    pt=pstart
    ni=-6.362+1.193*s1+0.00098*s1**2
    pw=np.zeros([int((pstop-pstart)/pint),3])
    for m in range(int((pstop-pstart)/pint)):
        pt=pstart+(m+1)*pint
        w=2*np.pi/pt
        tao=0.0
        p1=sum(rv*np.cos(w*(t-tao)))
        p3=sum(rv*np.sin(w*(t-tao)))
        p2=sum((np.cos(w*(t-tao)))**2)
        p4=sum((np.sin(w*(t-tao)))**2)
        px=0.5*(p1**2/p2+p3**2/p4)/sx2
        pw[m,0]=pt
        pw[m,1]=px
        pw[m,2]=1-(1-np.exp(-px))**ni
        p1=0
        p2=0
        p3=0
        p4=0
    fig=plt.figure()
    ax=fig.add_subplot(111)
    l1=ax.plot(pw[:,0],pw[:,1],'b',alpha=0.9,label='Power')
    ax.set_xlim([60,700])
    ax.set_xlabel('Period (day)',fontsize=16)
    
    ax2=ax.twinx()
    l2=ax2.plot(pw[:,0],pw[:,2],'y',alpha=0.9,label='False Alarm Probability')
    l=l1+l2
    labs=[x.get_label() for x in l]
    ax.legend(l,labs,bbox_to_anchor=(0.78,0.9))
    ax.set_title(title)
    ax.set_ylabel('Power',fontsize=16)
    ax.set_xticks([90,180,360,540])
    ax2.set_ylabel('False Alarm Probability',fontsize=16)
    plt.close

mearth = const.M_earth.cgs.value #grams
msun   = const.M_sun.cgs.value
au_per_day     = 1731460 #meters per second

mstar  = 1.034


def sampler_to_theta_max(run):
    samples = run.sampler.get_chain(flat=True, thin=run.thin)
    run.theta_max = samples[np.argmax(run.sampler.get_log_prob(flat=True, thin=run.thin))]


def bic(run):
    k = len(run.theta_max)
    n = 0
    for i in range(run.system.nplanets_ttvs):
        n += len(run.system.measured[i])
    n += len(run.system.rvbjd)
    l = np.argmax(run.sampler.get_log_prob(flat=True, thin=run.thin))
    bic = k * np.log(n) - (2*l)
    run.bic = bic