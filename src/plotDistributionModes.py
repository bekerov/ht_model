#!/usr/bin/env python

import sys
import os
import random
import logging
import numpy as np
import cPickle as pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from termcolor import colored
from scipy import stats

import simulationFunctions as sf
from loadTaskParams import *

# set up logging
logging.basicConfig(format='')
lgr = logging.getLogger("plotDistributionModes.py")
lgr.setLevel(level=logging.INFO)

if __name__=='__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    if not os.path.isfile("modes.pickle"):
        lgr.info("Generating modes.pickle file")
        n_actions_learned = np.zeros(n_trials)
        lgr.info("Loading dists.pickle file")
        with open("dists.pickle", "r") as dists_file:
            r1_dists = pickle.load(dists_file)
            r2_dists = pickle.load(dists_file)

        all_modes = list()

        for start_state in task_start_states_list:
            modes = np.zeros(len(r1_dists))
            for state_action_dist_idx in range(len(r1_dists)):
                r1_dist = r1_dists[state_action_dist_idx]
                r2_dist = r2_dists[state_action_dist_idx]
                for i in range(n_trials):
                    n_actions_learned[i] = sf.run_simulation(r1_dist, r2_dist, start_state)
                modes[state_action_dist_idx] = stats.mode(n_actions_learned)[0][0]

            best_indices = np.where(modes == modes.min())[0]
            all_modes.append(modes)
            lgr.info("%s", colored("Start State: %s" % str(start_state), 'yellow', attrs = ['bold']))
            lgr.info("%s", colored("Smallest mode: %d" % modes.min(), 'white', attrs = ['bold']))
            lgr.info("%s", colored("Distribution indices %s" % str(best_indices), 'white', attrs = ['bold']))

        lgr.info("Writing modes.pickle file")
        with open("modes.pickle", "wb") as modes_file:
            pickle.dump(all_modes, modes_file)

    lgr.info("Loading modes.pickle file")
    with open("modes.pickle", "r") as modes_file:
        all_modes = pickle.load(modes_file)

    for i in range(len(task_start_states_list)):
        plt.subplot(2, 2, i+1)
        plt.plot(range(len(all_modes[i])), all_modes[i], 'ro')
        l = all_modes[i].min()
        u = all_modes[i].max()
        d = (u-l)/8
        lb = l - d
        ub = u + d
        plt.axis([0, len(all_modes[i]), lb, ub])
        plt.title('Start State: %s' % str(task_start_states_list[i]))
        plt.xlabel('Distribution Number')
        plt.ylabel('Mode Value')
        plt.grid()

    plt.suptitle('Modes of various state action distributions for each start state for %d trials' % n_trials)
    plt.show()

