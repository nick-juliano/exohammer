---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: exo_env
    language: python
    name: python3
---

# Demo: Using Exohammer End-to-End

```python
import emcee
import json
import pandas as pd
import exohammer as exo

# Load measurement data
rv_df = pd.read_csv('./example_data/RV_kepler-36.csv')
k36b_ttv_df = pd.read_csv('./example_data/TTV_kepler-36b.csv')
k36c_ttv_df = pd.read_csv('./example_data/TTV_kepler-36c.csv')

# Load priors for orbital elements and stellar mass
with open('./example_data/orbital_elements.json', 'r') as f:
    orbital_elements = json.load(f)
    m_star = orbital_elements.pop('m_star')

orbital_elements
```

```python
kepler_36 = exo.planetary_system.PlanetarySystem(nplanets_ttvs=2, 
                                                 nplanets_rvs=3, 
                                                 orbital_elements=orbital_elements)

data = exo.data.Data(m_star)

data.add_ttv(planet_label = 'b',
             epoch = k36b_ttv_df['epoch'].astype(int).to_list(),
             measured = k36b_ttv_df['Transit Time'].to_list(),
             error = k36b_ttv_df['error'].to_list())

data.add_ttv(planet_label = 'c',
             epoch = k36c_ttv_df['epoch'].astype(int).to_list(),
             measured = k36c_ttv_df['Transit Time'].to_list(),
             error = k36c_ttv_df['error'].to_list())

data.add_rvs(bjds = rv_df['BJD'].to_list(),
             velocities= rv_df['mean_velocity'].to_list(),
             error = rv_df['velocity_error'].to_list())
run = exo.mcmc_run.MCMCRun(kepler_36, data)

```

```python
run.explore_iteratively(total_iterations=10000000, 
                        checkpoints=1000, 
                        burnin_factor=.2, 
                        thinning_factor=.001,
	                    moves=[(emcee.moves.DEMove(live_dangerously=True), 0.9), 
                               (emcee.moves.DESnookerMove(live_dangerously=True), 0.1)],
	                    verbose=True, 
                        tune=True, 
                        silent=True)
```
