#!/usr/bin/env python

import sys
import random
import logging
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import randomActionDistribution
import expertActionDistribution
import taskSetup as ts
import simulationFunctions as sf

# set logging level
logging.basicConfig(level=logging.ERROR, format='%(asctime)s-%(levelname)s: %(message)s')

# load task params from pickle file
task_params = ts.load_task_parameters()
task_states_dict = task_params[ts.TaskParams.task_states_dict]
task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]
feature_matrix = task_params[ts.TaskParams.feature_matrix]
expert_visited_states_set = task_params[ts.TaskParams.expert_visited_states_set]
expert_state_action_dict = task_params[ts.TaskParams.expert_state_action_dict]
n_episodes = task_params[ts.TaskParams.n_episodes]
time_per_step = task_params[ts.TaskParams.time_per_step]

if __name__=='__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_actions_expert = np.zeros(n_trials)
    n_actions_random = np.zeros(n_trials)
    for i in range(n_trials):
        start_state = random.choice(tuple(task_start_state_set))
        expert_state_action_distribution_dict = expertActionDistribution.compute_expert_state_action_distribution_dict()
        random_state_action_distribution_dict = randomActionDistribution.compute_random_state_action_distribution_dict()
        n_actions_expert[i] = sf.run_simulation(expert_state_action_distribution_dict, expert_state_action_distribution_dict, start_state)
        n_actions_random[i] = sf.run_simulation(random_state_action_distribution_dict, random_state_action_distribution_dict, start_state)
    print "Number of trials = ", n_trials
    print "Metric: Number of action per trial"
    print "********************************************************************************"
    print "                Expert Policy               Random Policy"
    print "********************************************************************************"
    print "Min:\t\t ",format(np.amin(n_actions_expert), '.3f'), "\t\t\t", format(np.amin(n_actions_random), '.3f')
    print "Max:\t\t ",format(np.amax(n_actions_expert), '.3f'), "\t\t\t", format(np.amax(n_actions_random), '.3f')
    print "Mean:\t\t ",format(np.mean(n_actions_expert), '.3f'), "\t\t\t", format(np.mean(n_actions_random), '.3f')
    print "Median:\t\t ",format(np.median(n_actions_expert), '.3f'), "\t\t\t", format(np.median(n_actions_random), '.3f')
    print "Var:\t\t ", format(np.var(n_actions_expert), '.3f'), "\t\t\t", format(np.var(n_actions_random), '.3f')
    print "Std:\t\t ", format(np.std(n_actions_expert), '.3f'), "\t\t\t", format(np.std(n_actions_random), '.3f')

hist, bin_edges = np.histogram(n_actions_expert, bins = 100)
plt.figure(1)
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.xlabel('Number of Actions')
plt.ylabel('Frequency of Actiosn')
plt.title('Histogram of action frequency for agents using expert policy')

hist, bin_edges = np.histogram(n_actions_random, bins = 100)
plt.figure(2)
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.xlabel('Number of Actions')
plt.ylabel('Frequency of Actiosn')
plt.title('Histogram of action frequency for agents using random policy')

plt.show()

# the histogram of the data
#n, bins, patches = plt.hist(n_actions_expert, 100, normed=False, facecolor='green', alpha=0.75)
#plt.show()

#Number of trials =  1000000
#Metric: Number of action per trial
#********************************************************************************
                #Expert Policy               Random Policy
#********************************************************************************
#Min:              10.000                        10.000
#Max:              36.000                        865966.000
#Mean:             16.244                        22.021
#Median:           16.000                        17.000
#Var:              19.969                        1228522.234
#Std:              4.469                         1108.387

