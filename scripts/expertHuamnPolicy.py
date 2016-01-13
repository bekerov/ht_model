#!/usr/bin/env python

import sys
import os
import glob
import random
import logging
import numpy as np

from pprint import pprint
from termcolor import colored

import taskSetup as ts
#import simulationFunctions as sf

"""This module creates a common policy for the box color sort task
   based on the most frequent actions taken by humans given a particular
   state
"""

#def get_common_policy(task_state_action_map, expert_state_action_map):
    #"""Function to extract the common policy given the state action mapping from human-human task
    #Returns:
        #dict: mapping state to the most frequent action of that particular state
    #"""
    #policy = dict()
    #for task_state, state_actions in task_state_action_map.items():
        #actions = dict()
        #for action in state_actions:
            #actions[action] = 1.0/len(state_actions)
        #policy[task_state] = actions

    #for task_state, expert_actions in expert_state_action_map.items():
        #actions = {k: float(v)/total for total in (sum(expert_actions.values()),) for k, v in expert_actions.items()}
        #policy[task_state] = actions
    #return policy

def compute_expert_action_distribution(task_state_action_dict, expert_state_action_dict):
    """Function to compute expert Q value based on the processed video files.
       Note: It is likely that the experts don't visit all the possible states in the state space, thus for states not visited by experts we assign an action based on a uniform distribution for that state. For states visited by experts, assign values for actions based on their frequency.
    """
    expert_action_distribution = dict()
    # intially setup the policy from a uniform distribution of actions for states
    for task_state_tup, actions_dict in task_state_action_dict.items():
        expert_actions = dict()
        for action in actions_dict:
            expert_actions[action] = 1.0/len(actions_dict)
        expert_action_distribution[task_state_tup] = expert_actions

    # overrite the states visited by the experts by the common actions taken
    for task_state_tup, expert_actions in expert_state_action_dict.items():
        actions_dict = {k: float(v)/total for total in (sum(expert_actions.values()),) for k, v in expert_actions.items()}
        expert_action_distribution[task_state_tup] = actions_dict

    return expert_action_distribution


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')
    np.set_printoptions(formatter={'float': '{: 1.3f}'.format}, threshold=np.nan)
    task_params = ts.load_task_parameters()
    task_states_set = task_params[ts.TaskParams.task_states_set]
    task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]
    expert_visited_states_set = task_params[ts.TaskParams.expert_visited_states_set]
    expert_state_action_dict = task_params[ts.TaskParams.expert_state_action_dict]
    expert_action_distribution = compute_expert_action_distribution(task_state_action_dict, expert_state_action_dict)
    pprint(expert_action_distribution)
    #print "Total number of actions by agents using expert policy is %d" % sf.run_simulation(get_common_policy(task_state_action_map, expert_state_action_map), get_common_policy(task_state_action_map, expert_state_action_map), random.choice(tuple(task_start_states)))

if __name__=='__main__':
    main()
