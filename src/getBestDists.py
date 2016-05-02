#!/usr/bin/env python

import sys
import logging
import random
import numpy as np
import cPickle as pickle

from termcolor import colored
from scipy import stats

import simulationFunctions as sf

from loadTaskParams import *

# set up logging
logging.basicConfig(format='')
lgr = logging.getLogger("getBestDists.py")
lgr.setLevel(level=logging.INFO)

MAX_BEST_STATE_ACTION_DISTS = 10

def getBestDists(n_trials = 100):
    lgr.info("Loading dists.pickle file")
    with open("dists.pickle", "r") as dists_file:
        r1_dists = pickle.load(dists_file)
        r2_dists = pickle.load(dists_file)

    n_actions_learned = np.zeros(n_trials)
    r1_best_dists = dict()
    r2_best_dists = dict()
    all_modes = list()
    lgr.info("%s", colored("Running simulation amongst %d state action distribution for getting the top 10 with lowest number of actions for various start states for %d trials" % (len(r1_dists), n_trials), 'white', attrs = ['bold']))
    for start_state in task_start_states_list:
        r1_best_dists[start_state] = list()
        r2_best_dists[start_state] = list()
        modes = np.zeros(len(r1_dists))
        for state_action_dist_idx in range(len(r1_dists)):
            r1_dist = r1_dists[state_action_dist_idx]
            r2_dist = r2_dists[state_action_dist_idx]
            for i in range(n_trials):
                n_actions_learned[i] = sf.run_simulation(r1_dist, r2_dist, start_state)
            modes[state_action_dist_idx] = stats.mode(n_actions_learned)[0][0]

        best_indices = np.where(modes == modes.min())[0]
        all_modes.append(modes)
        if len(best_indices) > MAX_BEST_STATE_ACTION_DISTS:
            best_indices = np.random.choice(best_indices, MAX_BEST_STATE_ACTION_DISTS, replace = False)

        lgr.info("%s", colored("Start State: %s" % str(start_state), 'yellow', attrs = ['bold']))
        lgr.info("%s", colored("Smallest mode: %d" % modes.min(), 'white', attrs = ['bold']))
        lgr.info("%s", colored("Distribution indices %s" % str(best_indices), 'white', attrs = ['bold']))

        for best_idx in np.nditer(best_indices):
            r1_best_dists[start_state].append(r1_dists[best_idx])
            r2_best_dists[start_state].append(r2_dists[best_idx])

    lgr.info("Writing the start state to best numpy distribution list dictionary to best_dists.pickle")
    with open("best_dists.pickle", "wb") as best_dists_file:
        pickle.dump(r1_best_dists, best_dists_file)
        pickle.dump(r2_best_dists, best_dists_file)

    lgr.info("Writing modes.pickle file")
    with open("modes.pickle", "wb") as modes_file:
        pickle.dump(all_modes, modes_file)
        pickle.dump(n_trials, modes_file)

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    getBestDists(n_trials)

