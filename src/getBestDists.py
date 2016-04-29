#!/usr/bin/env python

import sys
import logging
import random
import pprint
import numpy as np
import cPickle as pickle

from termcolor import colored
from scipy import stats

import qLearning as ql
import featureExpectation as mu
import simulationFunctions as sf

from loadTaskParams import *
from helperFuncs import *

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

    #t = r1_dists[0]
    #s = ts.State(n_r=2, n_h=0, t_r=0, t_h=0, b_r=1, b_h=0, e=0)
    #state_idx = task_states_list.index(s)
    #print task_states_list[state_idx], select_random_action(r1_dists[0][state_idx])
    n_actions_learned = np.zeros(n_trials)
    r1_best_dists = dict()
    r2_best_dists = dict()
    lgr.info("Running simulation amongst %d state action distribution for getting the top 10 with lowest number of actions for various start states", len(r1_dists))
    for start_state in task_start_states_list:
        r1_best_dists[start_state] = list()
        r2_best_dists[start_state] = list()
        modes = np.zeros(len(r1_dists))
        for state_action_dist_idx in range(len(r1_dists)):
            r1_dist = r1_dists[state_action_dist_idx]
            r2_dist = r2_dists[state_action_dist_idx]

            #r1_learned_state_action_distribution_dict = convert_to_dict_from_numpy(r1_dist)
            #r2_learned_state_action_distribution_dict = convert_to_dict_from_numpy(r2_dist)

            for i in range(n_trials):
                #n_actions_learned[i] = sf.run_simulation(r1_learned_state_action_distribution_dict, r2_learned_state_action_distribution_dict, start_state)
                n_actions_learned[i] = sf.run_simulation(r1_dist, r2_dist, start_state)
            modes[state_action_dist_idx] = stats.mode(n_actions_learned)[0][0]

            #assert(np.array_equal(r1_dist, convert_to_numpy_from_dict(r1_learned_state_action_distribution_dict)))
            #assert(np.array_equal(r2_dist, convert_to_numpy_from_dict(r2_learned_state_action_distribution_dict)))
        best_indices = np.where(modes == modes.min())[0]

        if len(best_indices) > MAX_BEST_STATE_ACTION_DISTS:
            best_indices = np.random.choice(best_indices, MAX_BEST_STATE_ACTION_DISTS, replace = False)

        lgr.info("%s", colored("Start State: %s" % str(start_state), 'yellow', attrs = ['bold']))
        lgr.info("%s", colored("Smallest mode: %d" % modes.min(), 'white', attrs = ['bold']))
        lgr.info("%s", colored("Policy indices %s" % str(best_indices), 'white', attrs = ['bold']))

        for best_idx in np.nditer(best_indices):
            r1_best_dists[start_state].append(r1_dists[best_idx])
            r2_best_dists[start_state].append(r2_dists[best_idx])

    lgr.info("Writing the start state to best numpy distribution list dictionary to best_dists.pickle")
    with open("best_dists.pickle", "wb") as best_dists_file:
        pickle.dump(r1_best_dists, best_dists_file)
        pickle.dump(r2_best_dists, best_dists_file)

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    getBestDists(n_trials)

