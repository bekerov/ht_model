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
lgr = logging.getLogger("extractPolicy.py")
lgr.setLevel(level=logging.INFO)

if __name__ == "__main__":
    with open("dists.pickle", "r") as dists_file:
        r1_dists = pickle.load(dists_file)
        r2_dists = pickle.load(dists_file)

    n_trials = 100
    n_actions_learned = np.zeros(n_trials)
    for start_state in task_start_states_list:
        modes = np.zeros(len(r1_dists))
        for state_action_dist_idx in range(len(r1_dists)):
            r1_dist = r1_dists[state_action_dist_idx]
            r2_dist = r2_dists[state_action_dist_idx]

            r1_learned_state_action_distribution_dict = convert_to_dict_from_numpy(r1_dist)
            r2_learned_state_action_distribution_dict = convert_to_dict_from_numpy(r2_dist)

            for i in range(n_trials):
                n_actions_learned[i] = sf.run_simulation(r1_learned_state_action_distribution_dict, r2_learned_state_action_distribution_dict, start_state)
            modes[state_action_dist_idx] = stats.mode(n_actions_learned)[0][0]

            assert(np.array_equal(r1_dist, convert_to_numpy_from_dict(r1_learned_state_action_distribution_dict)))
            assert(np.array_equal(r2_dist, convert_to_numpy_from_dict(r2_learned_state_action_distribution_dict)))
        best_indices = np.where(modes == modes.min())[0]
        lgr.info("%s", colored("Start State: %s" % str(start_state), 'yellow', attrs = ['bold']))
        lgr.info("%s", colored("Smallest mode: %d" % modes.min(), 'white', attrs = ['bold']))
        lgr.info("%s", colored("Policy indices %s" % str(best_indices), 'white', attrs = ['bold']))
    #t = extract_best_policy_dict_from_numpy(r1_dists[0])
    #q = convert_to_dict_from_numpy(r1_dists[1])
    #pprint.pprint(t)
    #pprint.pprint(q)
    #s = ts.State(n_r=3, n_h=0, t_r=0, t_h=1, b_r=1, b_h=0, e=0)
    #print t[s], q[s]
    #tidx = task_states_list.index(s)
    #print task_states_list[tidx], t[s], q[s], select_random_action(r1_dists[0][tidx])
    #print r1_dists[0][tidx]
    #n = convert_to_numpy_from_dict(q)
    #assert(np.array_equal(r1_dists[1], n))
