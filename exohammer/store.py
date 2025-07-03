"""
Run Storage and Restoration Module

Provides functionality for serializing and saving MCMC run results from exoplanet
modeling, including full run objects and derived CSVs, as well as restoring them
from disk.

Dependencies:
    - pickle
    - easygui
    - pandas
    - exohammer.planetary_system.PlanetarySystem
    - exohammer.data.Data
"""

from pickle import dump, load
from easygui import fileopenbox
from pandas import DataFrame

from exohammer.planetary_system import PlanetarySystem
from exohammer.data import Data


class StoreRun:
    """
    Container for saving and exporting the state of an exohammer MCMC run.

    Args:
        pkl_object (object): A run object with `sampler`, `system`, `theta_max`,
                             and other attributes defined during/after MCMC execution.

    Attributes:
        run (object): The input run object to be serialized or exported.
    """

    def __init__(self, pkl_object):
        self.run = pkl_object

    def serialize(self):
        """
        Serializes the entire run object to a `.obj` file using `pickle`.

        Output path is constructed from `self.run.output_path` and `self.run.date`.
        """
        filename = self.run.output_path + f"pickle_{self.run.date}.obj"
        with open(filename, 'wb') as file:
            dump(self, file)

    def store_csvs(self):
        """
        Serializes relevant components of the run and saves them as individual `.obj` and `.csv` files.

        Includes:
            - Planetary system and data objects
            - MCMC chain, labels, and log probabilities
            - TTV and RV model predictions
            - MCMC metadata
        """
        sys = self.run.system

        # Store the system definition as a standalone object
        planetary_system_obj = PlanetarySystem(sys.nplanets_ttvs, sys.nplanets_rvs, sys.orbital_elements, theta=None)
        with open(self.run.output_path + "pickle_planetary_system.obj", 'wb') as f:
            dump(planetary_system_obj, f)

        # Store the data object
        data_obj = Data(sys.m_star,
                        [sys.epoch, sys.measured, sys.error],
                        [sys.rvbjd, sys.rvmnvel, sys.rverrvel])
        with open(self.run.output_path + "pickle_data.obj", 'wb') as f:
            dump(data_obj, f)

        # Extract chain and log-probability info
        self.run.flatchain = self.run.sampler.get_chain(flat=True, discard=self.run.discard, thin=self.run.thin)
        self.run.log_prob = self.run.sampler.get_log_prob(flat=True, thin=self.run.thin)
        self.run.labels = self.run.system.variable_labels
        self.run.chain = self.run.sampler.get_chain(discard=self.run.discard, thin=self.run.thin)

        # Generate final model predictions
        self.run.ttv_model, self.run.ttv_epoch, self.run.rv_model = self.run.system.model(
            self.run.theta_max, self.run.system
        )

        # Write core data products to CSV
        DataFrame(self.run.flatchain).to_csv(self.run.output_path + 'flatchain.csv', index=False)
        DataFrame(self.run.labels).to_csv(self.run.output_path + 'labels.csv', index=False)
        DataFrame(self.run.ttv_model).to_csv(self.run.output_path + 'ttv_model.csv', index=False)
        DataFrame(self.run.ttv_epoch).to_csv(self.run.output_path + 'ttv_epoch.csv', index=False)
        DataFrame(self.run.rv_model).to_csv(self.run.output_path + 'rv_model.csv', index=False)

        # Write state and diagnostics
        DataFrame(self.run.pos).to_csv(self.run.output_path + 'pos.csv', index=False)
        DataFrame(self.run.prob).to_csv(self.run.output_path + 'prob.csv', index=False)
        DataFrame(self.run.state).to_csv(self.run.output_path + 'state.csv', index=False)
        DataFrame(self.run.log_prob).to_csv(self.run.output_path + 'logprob.csv', index=False)

        # Miscellaneous run configuration values
        misc = [
            self.run.nwalkers,
            self.run.niter,
            self.run.moves,
            self.run.burnin_factor,
            self.run.thinning_factor,
            self.run.discard,
            self.run.thin,
            self.run.tune,
            self.run.silent,
            self.run.total_iterations,
            self.run.theta_max
        ]
        DataFrame(misc).to_csv(self.run.output_path + 'misc.csv', index=False)


def restore(filename=None):
    """
    Restore a stored `StoreRun` object from a pickle file.

    Args:
        filename (str, optional): Full path to `.obj` pickle file.
                                  If None, GUI file selector is used.

    Returns:
        object: The restored `run` object.
    """
    if filename is None:
        filename = fileopenbox()

    with open(filename, 'rb') as file:
        run_wrapper = load(file)

    return run_wrapper.run
