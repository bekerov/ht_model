#!/usr/bin/env python

import sys
import pprint
import numpy as np
import cPickle as pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy import stats

import simulationFunctions as sf
from loadTaskParams import *


if __name__=='__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_actions_learned = np.zeros(n_trials)
    with open("agent_dists_dict.pickle", "r") as agent_dists_dict_file:
        r1_dists_dict = pickle.load(agent_dists_dict_file)
        r2_dists_dict = pickle.load(agent_dists_dict_file)

    all_modes = list()
    best_policy_list = list()

    for start_state in task_start_states_list:
        modes = list()
        for policy_idx in range(len(r1_dists_dict)):
            r1_learned_state_action_distribution_dict = r1_dists_dict[policy_idx]
            r2_learned_state_action_distribution_dict = r2_dists_dict[policy_idx]

            for i in range(n_trials):
                n_actions_learned[i] = sf.run_simulation(r1_learned_state_action_distribution_dict, r2_learned_state_action_distribution_dict, start_state)
            modes.append(stats.mode(n_actions_learned)[0][0])
        all_modes.append(modes)
        smallest, policy_indices = locate_min(modes)
        best_policy_list.append(policy_indices)
        print "Start State: ", start_state
        print "Smallest mode: ", smallest
        print "Policy indices", policy_indices

    best_policy_indices = set([idx for policy_list in best_policy_list for idx in policy_list])
    print len(r1_dists_dict), len(best_policy_indices)


    for i in range(len(task_start_states_list)):
        plt.subplot(2, 2, i+1)
        plt.plot(range(len(r1_dists_dict)), all_modes[i], 'ro')
        l = min(all_modes[i])
        u = max(all_modes[i])
        d = (u-l)/8
        lb = l - d
        ub = u + d
        plt.axis([0, len(r1_dists_dict), lb, ub])
        plt.title('Start State: %s' % str(task_start_states_list[i]))
        plt.xlabel('Policy Number')
        plt.ylabel('Mode Value')
        plt.grid()

    plt.suptitle('Modes of various policy for each start state for %d trials' % n_trials)
    plt.show()

