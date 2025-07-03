# Exohammer Documentation

Welcome to the **Exohammer** documentation — a Python library for Bayesian analysis of exoplanetary systems using TTV and RV data.

---

## 📦 Installation

```bash
pip install exohammer
```

---

## 🚀 Quickstart

```python
import exohammer as exo
import emcee

# Define your planetary system and data
kepler_xx = exo.planetary_system.PlanetarySystem(...)
data = exo.data.Data(...)

# Run MCMC
run = exo.mcmc_run.MCMCRun(kepler_xx, data)
run.explore_iteratively(...)
```

---

## 📚 Modules

### `planetary_system`
Defines and validates orbital elements for a system.

### `data`
Loads and stores TTV and RV datasets.

### `mcmc_run`
Handles setup and execution of MCMC chains.

### `store`
Serializes, saves, and reloads MCMC runs.

### `utilities`
Utility functions for plotting, thinning, and diagnostics.

---

## 📈 Plotting

```python
run.plot_ttvs()
run.plot_rvs()
run.plot_chains()
run.plot_corner()
```

---

## 💾 Saving Results

```python
store = exo.store.StoreRun(run)
store.store_csvs()
store.serialize()
```

---

## 🧪 Development

To build the package locally:

```bash
pip install -e .
```

To build documentation:

```bash
cd docs
make html
```

To deploy docs:

```bash
bash deploy_docs.sh
```

---

## 🔗 Citations

- [TTVFast](https://github.com/mindriot101/ttvfast-python) — Deck et al. (2014)
- [emcee](https://github.com/dfm/emcee) — Foreman-Mackey et al. (2013)

---

For more, visit the [GitHub Repo](https://github.com/nick-juliano/exohammer)

Contact: Nick Juliano — nick_juliano@icloud.com
