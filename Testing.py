#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:19:34 2021

@author: nickjuliano
"""

import exohammer as exo
from Input_Measurements import *
from Input_Measurements import rv_102021 as rv
import tracemalloc
import gc
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type, cumulative=True)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

tracemalloc.start()

# ... run your application ...



# ... run your application ...


print(orbital_elements_4body)

kepler_36 = exo.planetary_system.planetary_system(2, 3, orbital_elements_4body, theta=None)
data = exo.data.data(mstar, [epoch, measured, error], rv)
run = exo.mcmc_run.mcmc_run(kepler_36, data)

# lnprob=exo.prob_functions.lnprob(params, system)
# print(lnprob)
niter_total=10000
chopper=10
chopped=int(niter_total/chopper)
run.explore(chopped, thin=1, verbose=True)

snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

top_stats = snapshot.statistics('lineno', cumulative=True)

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
for i in range(chopper):
	#print(str(i))
	run.explore_again(chopped, verbose=True)
	if i%1==0:

		#store = exo.store.store_run(run)
		#store.store()

		#run.plot_chains()
		#run.plot_rvs()
		#run.plot_ttvs()
		#run.autocorr()
		#run.summarize()
		#run.plot_corner()

		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
		top_stats = snapshot.statistics('lineno', cumulative=True)
		print("[ Top 10 ]")
		for stat in top_stats[:10]:
			print(stat)
		gc.collect()

print(run.theta_max)

store = exo.store.store_run(run)
store.store()

run.plot_chains()
run.plot_rvs()
run.plot_ttvs()
run.autocorr()
run.summarize()
run.plot_corner()

