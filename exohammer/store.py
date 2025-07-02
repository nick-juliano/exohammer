# -*- coding: utf-8 -*-

from pickle import dump, load
from easygui import fileopenbox
from pandas import DataFrame

from exohammer.planetary_system import PlanetarySystem
from exohammer.data import Data

class StoreRun:
    def __init__(self, pkl_object):
        self.run = pkl_object

    def serialize(self):
        pkl_object = self
        filename = self.run.output_path + "pickle_" + self.run.date + '.obj'
        file = open(filename, 'wb')
        dump(pkl_object, file)

    def store_csvs(self):
        sys=self.run.system
        pkl_object_planetary_system = PlanetarySystem(sys.nplanets_ttvs,
                                                      sys.nplanets_rvs,
                                                      sys.orbital_elements,
                                                      theta=None)
        filename = self.run.output_path + "pickle_planetary_system.obj"
        file = open(filename, 'wb')
        dump(pkl_object_planetary_system, file)
        file.close()

        pkl_object_data = Data(sys.m_star,
                               [sys.epoch, sys.measured, sys.error],
                               [sys.rvbjd, sys.rvmnvel, sys.rverrvel])
        filename = self.run.output_path + "pickle_data.obj"
        file = open(filename, 'wb')
        dump(pkl_object_data, file)
        file.close()

        self.run.flatchain = self.run.sampler.get_chain(flat=True, discard=self.run.discard, thin=self.run.thin)
        self.run.log_prob = self.run.sampler.get_log_prob(flat=True, thin=self.run.thin)
        self.run.labels = self.run.system.variable_labels
        self.run.chain = self.run.sampler.get_chain(discard=self.run.discard, thin=self.run.thin)
        self.run.ttv_model, self.run.ttv_epoch, self.run.rv_model = self.run.system.model(self.run.theta_max, self.run.system)

        DataFrame(self.run.flatchain).to_csv(self.run.output_path + 'flatchain.csv')
        DataFrame(self.run.labels).to_csv(self.run.output_path + 'labels.csv')
        DataFrame(self.run.ttv_model).to_csv(self.run.output_path + 'ttv_model.csv')
        DataFrame(self.run.ttv_epoch).to_csv(self.run.output_path + 'ttv_epoch.csv')
        DataFrame(self.run.rv_model).to_csv(self.run.output_path + 'rv_model.csv')
        DataFrame(self.run.pos).to_csv(self.run.output_path + 'pos.csv')
        DataFrame(self.run.prob).to_csv(self.run.output_path + 'prob.csv')
        DataFrame(self.run.state).to_csv(self.run.output_path + 'state.csv')
        DataFrame(self.run.log_prob).to_csv(self.run.output_path + 'logprob.csv')

        run=self.run
        misc = [run.nwalkers, run.niter, run.moves, run.burnin_factor, run.thinning_factor, run.discard, run.thin,
                run.tune, run.silent, run.total_iterations, run.theta_max]
        DataFrame(misc).to_csv(self.run.output_path + 'misc.csv')

def restore(filename=None):
    if filename is None:
        filename = fileopenbox()
        file = open(filename, 'rb')
        run = load(file)
    else:
        file = open(filename, 'rb')
        run = load(file)
    return run.run
