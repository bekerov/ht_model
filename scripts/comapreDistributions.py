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

import expertActionDistribution as ex
import simulationFunctions as sf

from helperFuncs import compute_random_state_action_distribution
from loadTaskParams import *

logging.basicConfig(format='')
lgr = logging.getLogger("compareDistributions.py")
lgr.setLevel(level=logging.INFO)

if __name__=='__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_actions_expert = np.zeros(n_trials)
    n_actions_random = np.zeros(n_trials)
    n_actions_learned = np.zeros(n_trials)
    lgr.info("Loading best_dists.pickle file")
    with open("../pickles/best_dists.pickle", "r") as best_dists_file:
        r1_best_dists = pickle.load(best_dists_file)
        r2_best_dists = pickle.load(best_dists_file)

    for start_state in task_start_states_list:
        r1_best_dist = random.choice(r1_best_dists[start_state])
        r2_best_dist = random.choice(r2_best_dists[start_state])

        for i in range(n_trials):
            expert_state_action_distribution = ex.compute_expert_state_action_distribution()
            n_actions_expert[i] = sf.run_simulation(expert_state_action_distribution, expert_state_action_distribution, start_state)

            random_state_action_distribution = compute_random_state_action_distribution()
            n_actions_random[i] = sf.run_simulation(random_state_action_distribution, random_state_action_distribution, start_state)

            n_actions_learned[i] = sf.run_simulation(r1_best_dist, r2_best_dist, start_state)
        lgr.info("%s", colored("Number of trials = %d" % n_trials, 'white', attrs = ['bold']))
        lgr.info("%s", colored("Metric: Number of actions per trial", 'white', attrs = ['bold']))
        lgr.info("%s", colored("Start State: %s" % str(start_state), 'magenta', attrs = ['bold']))
        lgr.info("%s", colored("************************************************************************************************************", 'white', attrs = ['bold']))
        lgr.info("%s%s%s", colored("                Expert Policy            ", 'red', attrs = ['bold']), colored("Learned Policy        ", 'green', attrs = ['bold']), colored("Random Policy", 'blue', attrs = ['bold']))
        lgr.info("%s", colored("************************************************************************************************************", 'white', attrs = ['bold']))
        lgr.info("%s\t\t%s\t\t\t%s\t\t\t%s", colored("MIN:", 'white', attrs = ['bold']), colored("%s" % format(np.amin(n_actions_expert), '.3f'), 'red', attrs = ['bold']), colored("%s" % format(np.amin(n_actions_learned), '.3f'), 'green', attrs = ['bold']), colored("%s" % format(np.amin(n_actions_random), '.3f'), 'blue', attrs = ['bold']))
        lgr.info("%s\t\t%s\t\t\t%s\t\t\t%s", colored("MAX:", 'white', attrs = ['bold']), colored("%s" % format(np.amax(n_actions_expert), '.3f'), 'red', attrs = ['bold']), colored("%s" % format(np.amax(n_actions_learned), '.3f'), 'green', attrs = ['bold']), colored("%s" % format(np.amax(n_actions_random), '.3f'), 'blue', attrs = ['bold']))
        lgr.info("%s\t\t%s\t\t\t%s\t\t\t%s", colored("MEAN:", 'white', attrs = ['bold']), colored("%s" % format(np.mean(n_actions_expert), '.3f'), 'red', attrs = ['bold']), colored("%s" % format(np.mean(n_actions_learned), '.3f'), 'green', attrs = ['bold']), colored("%s" % format(np.mean(n_actions_random), '.3f'), 'blue', attrs = ['bold']))
        lgr.info("%s\t\t%s\t\t\t%s\t\t\t%s", colored("MODE:", 'white', attrs = ['bold']), colored("%s" % format(stats.mode(n_actions_expert)[0][0], '.3f'), 'red', attrs = ['bold']), colored("%s" % format(stats.mode(n_actions_learned)[0][0], '.3f'), 'green', attrs = ['bold']), colored("%s" % format(stats.mode(n_actions_random)[0][0], '.3f'), 'blue', attrs = ['bold']))
        lgr.info("%s\t\t%s\t\t\t%s\t\t\t%s", colored("MEDIAN:", 'white', attrs = ['bold']), colored("%s" % format(np.median(n_actions_expert), '.3f'), 'red', attrs = ['bold']), colored("%s" % format(np.median(n_actions_learned), '.3f'), 'green', attrs = ['bold']), colored("%s" % format(np.median(n_actions_random), '.3f'), 'blue', attrs = ['bold']))
        lgr.info("%s\t\t%s\t\t\t%s\t\t\t%s", colored("VAR:", 'white', attrs = ['bold']), colored("%s" % format(np.var(n_actions_expert), '.3f'), 'red', attrs = ['bold']), colored("%s" % format(np.var(n_actions_learned), '.3f'), 'green', attrs = ['bold']), colored("%s" % format(np.var(n_actions_random), '.3f'), 'blue', attrs = ['bold']))
        lgr.info("%s\t\t%s\t\t\t%s\t\t\t%s", colored("STD:", 'white', attrs = ['bold']), colored("%s" % format(np.std(n_actions_expert), '.3f'), 'red', attrs = ['bold']), colored("%s" % format(np.std(n_actions_learned), '.3f'), 'green', attrs = ['bold']), colored("%s" % format(np.std(n_actions_random), '.3f'), 'blue', attrs = ['bold']))
        user_input = raw_input('Press Enter to continue, Q-Enter to quit')
        if user_input.upper() == 'Q':
            break

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

