![image](https://github.com/user-attachments/assets/bb36a18a-e5cb-4ad5-b11c-831d2a3e92f6)


# exohammer
#### The Exoplanet MCMC Hammer
___


_A python library designed to provide utility for performing bayesian analysis on n-body planetary systems with any combination of TTV and RV measurements._




## Installation
___
exohammer is pip installable:

`pip install exohammer`

If you prefer to install from source code you can follow these steps:
Once you clone/fork/download/obtain this repo on your local machine, you can install the library by following these steps:
- cd into the directory
- run `pip install .`


## Usage
___

- Define *RV* and *TTV* nested lists in the following form:
  - **TTV**: `TTV = [[epoch_b, epoch_c], [TT_b,TT_c], [error_b, error_c]]`
    - ...and so on for each additional planet
  - **RV**: `RV = [[BJD], [mean_velocity], [velocity_error]]`

- Define your orbital elements (mass, period, eccentricity, inclination, longitude of ascending node, argument of periastron, mean anomaly) in dictionary form.
  - each key-value pair should be of the form "element_b" : [value]
  - regarding each value:
    - [value] denotes a fixed element
    - [minimum, maximum] denotes a flat prior with hard bounds
    - [mu, sigma, 0] denotes a gaussian prior
- Pass your orbital elements into a `PlanetarySystem` instance along with the number of planets you are fitting to the included data.
- Pass your TTVs, RVs, and orbital elements into a `Data` instance along with the mass of the host star in units of m_sun
- Pass your `PlanetarySystem` and `Data` instances into an `MCMCRun` instance and advance the chain iteratively using the MCMCRun.explore_iteratively() method

Toy Example:

```python
import exohammer as exo
import emcee
# Define TTV Parameters

epoch_b = [0, 1, 3, 4]
tt_b = [24500.10, 245010.32, 245029.97, 245041.26]
error_b = [0.02, 0.03, 0.028, 0.012]
## repeat for all planets, then combine as below
ttv = [[epoch_b, epoch_c], [TT_b, TT_c], [error_b, error_c]]

# Define RV Parameters

rv_bjd = [2456138.960111, 2456202.74719, 2456880.986613],
rv_mnvel = [2.26, -4.6378, -6.48],
rv_err = [3.21964406967163, 4.39822244644165, 4.03610754013062]
## combine as below
rv = [rv_bjd, rv_mnvel, rv_err]

# Define orbital elements

orbital_elements = {'mass_b' : [.000001], # m_sun, fixed value
                    'period_b' : [13.849 - .05, 13.849 + .05], # Days, flat prior  
                    'eccentricity_b' : [0.00, 0.04, 0.01], # gaussian prior
                    'inclination_b' : [89.5, 90.5], # Degrees
                    'longnode_b' : [-45.0, 45.0], # Degrees
                    'argument_b' : [0.0, 360.0], # Degrees
                    'mean_anomaly_b' : [0.0, 360.0], # Degrees

                    'mass_c'         : [.000001], # fixed value
                    'period_c'       : [16.0 - .05, 16.0 + .05], # flat prior           
                    'eccentricity_c' : [0.00, 0.04, 0], # gaussian prior
                    'inclination_c'  : [89.5, 90.5],
                    'longnode_c'     : [0.0],
                    'argument_c'     : [0.0, 360.0],
                    'mean_anomaly_c' : [0.0, 360.0]}

# Pass orbital elements into planetary system instance
kepler_xx = exo.planetary_system.PlanetarySystem(nplanets_ttvs=2, 
                                                 nplanets_rvs=2, 
                                                 orbital_elements=orbital_elements, 
                                                 theta=None)
                                                
data = exo.data.Data(mstar, [epoch, measured, error], rv)

# Pass TTVs and RVs into a `Data` instance
mstar = 1.2 # m_sun
data = exo.data.Data(mstar, ttv, rv)

# Instantiate MCMCRun instance and perform fit
run = exo.mcmc_run.MCMCRun(kepler_xx, data)

run.explore_iteratively(total_iterations=100000000, #The total number of steps to advance your chains
                        checkpoints=10000, #At this number of steps, the run will save your run for incremental evaluation
                        burnin_factor=.2, #Percentage of the run to discard as burn in.
                        thinning_factor=.0001, #Percentage of the run to thin the entire run by
	                    moves=[(emcee.moves.DEMove(live_dangerously=True), 0.9), (emcee.moves.DESnookerMove(live_dangerously=True), 0.1),], #See https://emcee.readthedocs.io/en/stable/tutorials/moves/
	                    verbose=False, #Show progress bar
                        tune=True, #Passes to emcee.EnsembleSampler.sample. 'True' recommended
                        silent=True, #Whether to display plots directly in the IDE
                        )
```


### Additional Utilities

### To review the success of the run
```python
run.plot_rvs()    # plots the RVs of the posterior planetary system 
                  # and compares it to the input data
run.plot_ttvs()   # plots the TTVs of the posterior planetary system 
                  # and compares it to the input data
run.autocorr()     # determines the autocorrelation of each parameter
run.plot_corner()  # plots the parameters in a corner plot
run.plot_chains()  # plots the path of each parameter
```
#### Storing

```python
# Storing
store = exo.store.StoreRun(run)
store.store_csvs() #stores Data instance (serialized) and most other attributes as csv files
store.serialize() #stores entire run in serialized format (very large)
store.restore() #reloades the serialized and stored output of store.serialize()

# Note that the save directory and naming is automatic, determined by initialization of MCMCRun. 
# Work to give saving/naming flexibility is ongoing. 

```

Every `checkpoint` steps of an `explore_iteratively()` MCMC run, `store_csvs()` is executed.

Note that the save directory and naming is automatic, determined by initialization of `MCMCRun`. Work to give saving/naming flexibility is ongoing. 




These methods are automatically executed every `checkpoint` steps of an `explore_iteratively` MCMC run.


___
Please see the code/docstrings for further information. Bugs, recommendations, and questions can be directed to me directly (nick_juliano@icloud.com).

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



