#!/usr/bin/env python

import sys
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

logging.basicConfig(format='')
lgr = logging.getLogger("policyComparision.py")
lgr.setLevel(level=logging.INFO)

if __name__=='__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_actions_learned = np.zeros(n_trials)
    with open("agent_policies.pickle", "r") as agent_policies_file:
        r1_policies = pickle.load(agent_policies_file)
        r2_policies = pickle.load(agent_policies_file)

    all_modes = list()

    for start_state in task_start_state_set:
        modes = list()
        for policy_idx in range(len(r1_policies)):
            r1_learned_state_action_distribution_dict = r1_policies[policy_idx]
            r2_learned_state_action_distribution_dict = r2_policies[policy_idx]

            for i in range(n_trials):
                n_actions_learned[i] = sf.run_simulation(r1_learned_state_action_distribution_dict, r2_learned_state_action_distribution_dict, start_state)
            modes.append(stats.mode(n_actions_learned)[0][0])
        all_modes.append(modes)

    plt.subplot(2, 2, 1)
    plt.plot(all_modes[0], range(len(r1_policies)), 'ro')
    plt.axis([0, len(r1_policies), 0, max(all_modes[0])])
    plt.title('Start State: %s' % str(task_start_state_set[0]))
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(all_modes[1], range(len(r1_policies)), 'ro')
    plt.axis([0, len(r1_policies), 0, max(all_modes[0])])
    plt.title('Start State: %s' % str(task_start_state_set[1]))
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(all_modes[2], range(len(r1_policies)), 'ro')
    plt.axis([0, len(r1_policies), 0, max(all_modes[0])])
    plt.title('Start State: %s' % str(task_start_state_set[2]))
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(all_modes[3], range(len(r1_policies)), 'ro')
    plt.axis([0, len(r1_policies), 0, max(all_modes[0])])
    plt.title('Start State: %s' % str(task_start_state_set[3]))
    plt.grid()

    plt.xlabel('Policy Number')
    plt.ylabel('Mode Value')
    plt.show()

# hist, bin_edges = np.histogram(n_actions_expert, bins = 100)
# plt.figure(1)
# plt.bar(bin_edges[:-1], hist, width = 1)
# plt.xlim(min(bin_edges), max(bin_edges))
# plt.xlabel('Number of Actions')
# plt.ylabel('Frequency of Actiosn')
# plt.title('Histogram of action frequency for agents using expert policy')

# hist, bin_edges = np.histogram(n_actions_random, bins = 100)
# plt.figure(2)
# plt.bar(bin_edges[:-1], hist, width = 1)
# plt.xlim(min(bin_edges), max(bin_edges))
# plt.xlabel('Number of Actions')
# plt.ylabel('Frequency of Actiosn')
# plt.title('Histogram of action frequency for agents using random policy')

# hist, bin_edges = np.histogram(n_actions_learned, bins = 100)
# plt.figure(3)
# plt.bar(bin_edges[:-1], hist, width = 1)
# plt.xlim(min(bin_edges), max(bin_edges))
# plt.xlabel('Number of Actions')
# plt.ylabel('Frequency of Actiosn')
# plt.title('Histogram of action frequency for agents using learned policy')

# plt.show()

# the histogram of the data
#n, bins, patches = plt.hist(n_actions_expert, 100, normed=False, facecolor='green', alpha=0.75)
#plt.show()

