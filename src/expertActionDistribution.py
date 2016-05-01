#!/usr/bin/env python

import sys
import random
import logging

import simulationFunctions as sf

from helperFuncs import convert_to_numpy_from_dict
from loadTaskParams import *

"""This module creates an expert action distribution for the box color sort task
   based on the most frequent actions taken by humans given a particular
   state.
"""

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
lgr = logging.getLogger("expertActionDistribution.py")
lgr.setLevel(level=logging.INFO)

def compute_expert_state_action_distribution():
    """Function to compute expert Q value based on the processed video files.
       Note: It is likely that the experts don't visit all the possible states in the state space, thus for states not visited by experts we assign an action based on a uniform distribution for that state. For states visited by experts, assign values for actions based on their frequency.
    """
    expert_state_action_distribution_dict = dict()
    # intially setup the policy from a uniform distribution of actions for states
    for task_state_tup, actions_dict in task_state_action_dict.items():
        expert_actions = dict()
        for action in actions_dict:
            expert_actions[action] = 1.0/len(actions_dict)
        expert_state_action_distribution_dict[task_state_tup] = expert_actions

    # overrite the states visited by the experts by the common actions taken
    for task_state_tup, expert_actions in expert_state_action_dict.items():
        actions_dict = {k: float(v)/total for total in (sum(expert_actions.values()),) for k, v in expert_actions.items()}
        expert_state_action_distribution_dict[task_state_tup] = actions_dict

    return convert_to_numpy_from_dict(expert_state_action_distribution_dict)


def simulate_expert_state_action_distribution():
    r1_dist = compute_expert_state_action_distribution()
    r2_dist = compute_expert_state_action_distribution()
    start_state = task_start_states_list[3]
    n_actions = sf.run_simulation(r1_dist, r2_dist, start_state)
    lgr.debug("Total number of actions by agents using expert policy is %d" % n_actions)
    return n_actions

if __name__=='__main__':
    total_actions = 0
    n_trials = int(sys.argv[1]) if len(sys.argv) == 2 else 100
    for i in range(n_trials):
        total_actions = total_actions + simulate_expert_state_action_distribution()

    lgr.info("average_actions = %0.2f", float(total_actions)/n_trials)

