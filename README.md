# exohammer
#### The Exoplanet MCMC Hammer
___


_A python module designed to perform bayesian analysis on n-body systems with any combination of TTV and RV measurements._




## Usage
___

- Define *RV* and *TTV* nested lists in the following form:
  - **TTV**: `TTV = [[epoch_b, epoch_c], [TT_b,TT_c], [error_b, error_c]]`
    - ...and so on for each additional planet
  - **RV**: `RV = [[data], [mean_velocity], [velocity_error]]`

- Define your orbital elements (mass, period, eccentricity, inclination, longitude of ascending node, argument of periastron, mean anomaly) in dictionary form.
  - each key-value pair should be of the form "element_b" : [value]
  - regarding each value:
    - [value] denotes a fixed element
    - [minimum, maximum] denotes a flat prior with hard bounds
    - [mu, sigma, 0] denotes a gaussian prior
    

```python
orbital_elements = {'mass_b'         : [.000001],                           # M_sun
                    'period_b'       : [13.849-.05,   13.849+.05       ],   # Days       
                    'eccentricity_b' : [0.00,         0.04             ],   
                    'inclination_b'  : [89.5,         90.5             ],   # Degrees
                    'longnode_b'     : [-45.0,        45.0             ],   # Degrees
                    'argument_b'     : [0.0,          360.0            ],   # Degrees
                    'mean_anomaly_b' : [0.0,          360.0            ],   # Degrees
     
                    'mass_c'         : [.000001],                           # fixed value
                    'period_c'       : [16.0 - .05,  16.0 + .05        ],   # flat prior           
                    'eccentricity_c' : [0.00, 0.04, 0                  ],   # gaussian prior
                    'inclination_c'  : [89.5,         90.5             ],
                    'longnode_c'     : [0.0                            ],
                    'argument_c'     : [0.0,          360.0            ],
                    'mean_anomaly_c' : [0.0,          360.0            ]}
```


- Pass your orbital elements into a `planetary system` instance along with the number of planets you are fitting to the included data.
- Pass your TTVs, RVs, and orbital elements into a `data` instance along with the mass of the host star in units of m_sun

- Pass your `planetary system` and `data` instances into an `mcmc_run` instance and advance the chain iteratively using the mcmc_run.explore() function

___
## Tutorial

### To perform an MCMC run

```python
import exohammer as exo
from Input_Measurements import rv, ttv, mstar, orbital_elements
# note that Input_Measurements holds the user-defined data and priors
# rv, ttv, mstar, and orbital_elements can also be defined here

planetary_system = exo.planetary_system.planetary_system(2, orbital_elements)
data = exo.data.data(mstar, ttv, rv, orbital_elements)

run = exo.mcmc_run.mcmc_run(planetary_system, data)
run.explore(100000, thin=10, verbose=True)
```

### To store the run for later analysis
```python
store = exo.store.store_run(run)
store.store()
```

### To plot the RVs and TTVs
```python
run.plot_rvs()    # plots the RVs of the posterior planetary system 
                  # and compares it to the input data
                  
run.plot_ttvs()   # plots the TTVs of the posterior planetary system 
                  # and compares it to the input data
```
### To review the success of the run
```python
run.autocorr()     # determines the autocorrelation of each parameter
run.plot_corner()  # plots the parameters in a corner plot
run.plot_chains()  # plots the path of each parameter
```
___
Please see the documentation for further information. Bugs, recommendations, and questions can be directed to me directly (nick_juliano@icloud.com).

## Citations
___
### TTVFast
This code relies extensively on _TTVFast_ (<https://github.com/mindriot101/ttvfast-python>) for modeling planetary systems. If you use this code, please cite:

Deck, Agol, Holman, & Nesvorny (2014), ApJ, 787, 132, arXiv:1403.1895.

-Katherine Deck, Eric Agol, Matt Holman, & David Nesvorny

### Emcee
This code also relies heavily on Emcee (<https://github.com/dfm/emcee>) for ensemble sampling.

Please cite Foreman-Mackey, Hogg, Lang & Goodman (2012)
(<https://arxiv.org/abs/1202.3665>) if you find this code useful in your
research. The BibTeX entry for the paper is::

    @article{emcee,
       author = {{Foreman-Mackey}, D. and {Hogg}, D.~W. and {Lang}, D. and {Goodman}, J.},
        title = {emcee: The MCMC Hammer},
      journal = {PASP},
         year = 2013,
       volume = 125,
        pages = {306-312},
       eprint = {1202.3665},
          doi = {10.1086/670067}
    }

___
