#!/usr/bin/env python

import sys
import logging
import random
import pprint
import cPickle as pickle

import simulationFunctions as sf

from helperFuncs import compute_random_state_action_distribution
from loadTaskParams import *

"""This module creates an random action distribution for the box color sort task.
"""
logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
lgr = logging.getLogger("randomActionDistribution.py")
lgr.setLevel(level=logging.INFO)

def simulate_random_state_action_distribution():
    """Function to simulate a random action distribution for both the agents
    """
    r1_dist = compute_random_state_action_distribution()
    r2_dist = compute_random_state_action_distribution()
    start_state = task_start_states_list[0]
    n_actions = sf.run_simulation(r1_dist, r2_dist, start_state)
    lgr.debug("Total number of actions by agents using expert policy is %d" % n_actions)
    return n_actions

if __name__=='__main__':
    total_actions = 0
    n_trials = int(sys.argv[1]) if len(sys.argv) == 2 else 100
    for i in range(n_trials):
        total_actions = total_actions + simulate_random_state_action_distribution()

    lgr.info("average_actions = %0.2f", float(total_actions)/n_trials)

