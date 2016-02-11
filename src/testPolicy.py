#!/usr/bin/env python

import sys
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
    with open("agent_policies.pickle", "r") as agent_policies_file:
        r1_policies = pickle.load(agent_policies_file)
        r2_policies = pickle.load(agent_policies_file)

    all_modes = list()

    for start_state in task_start_state_set:
        start_state = task_start_state_set[0]
        modes = list()
        for policy_idx in range(len(r1_policies)):
            r1_learned_state_action_distribution_dict = r1_policies[policy_idx]
            r2_learned_state_action_distribution_dict = r2_policies[policy_idx]

            for i in range(n_trials):
                n_actions_learned[i] = sf.run_simulation(r1_learned_state_action_distribution_dict, r2_learned_state_action_distribution_dict, start_state)
            modes.append(stats.mode(n_actions_learned)[0][0])
        all_modes.append(modes)

    for i in range(len(task_start_state_set)):
        plt.subplot(2, 2, i+1)
        plt.plot(range(len(r1_policies)), all_modes[i], 'ro')
        l = min(all_modes[i])
        u = max(all_modes[i])
        d = (u-l)/8
        lb = l - d
        ub = u + d
        plt.axis([0, len(r1_policies), lb, ub])
        plt.title('Start State: %s' % str(task_start_state_set[i]))
        plt.xlabel('Policy Number')
        plt.ylabel('Mode Value')
        plt.grid()

    plt.suptitle('Modes of various policy for each start state for %d trials' % n_trials)
    plt.show()

