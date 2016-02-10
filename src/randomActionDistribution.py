#!/usr/bin/env python

import logging
import random
import pprint
import cPickle as pickle

import simulationFunctions as sf

from loadTaskParams import *

"""This module creates an random action distribution for the box color sort task.
"""
logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
lgr = logging.getLogger("expertActionDistribution.py")
lgr.setLevel(level=logging.WARN)

def compute_random_state_action_distribution_dict():
    """Function to compute a random distribution for actions for each task state
    """
    random_state_action_distribution_dict = dict()
    for task_state_tup, actions_dict in task_state_action_dict.items():
        random_actions = dict()
        for action in actions_dict:
            random_actions[action] = random.random()
        random_actions = {k: float(v)/total for total in (sum(random_actions.values()),) for k, v in random_actions.items()}
        random_state_action_distribution_dict[task_state_tup] = random_actions

    return random_state_action_distribution_dict

def simulate_random_state_action_distribution():
    """Function to simulate a random action distribution for both the agents
    """
    r1_state_action_distribution_dict = compute_random_state_action_distribution_dict()
    r2_state_action_distribution_dict = compute_random_state_action_distribution_dict()
    start_state = task_start_state_set[0]
    n_actions = sf.run_simulation(r1_state_action_distribution_dict,r2_state_action_distribution_dict, start_state)
    lgr.info("Total number of actions by agents using expert policy is %d" % n_actions)
    return n_actions

if __name__=='__main__':
    total_actions = 0
    n_trials = 1
    for i in range(n_trials):
        total_actions = total_actions + simulate_random_state_action_distribution()

    print "average_actions = ", total_actions/n_trials


