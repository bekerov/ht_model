#!/usr/bin/env python

import sys
import os
import glob
import random
import logging
import numpy as np

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

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')
    np.set_printoptions(formatter={'float': '{: 1.3f}'.format}, threshold=np.nan)
    task_params = ts.load_task_parameters()
    #task_states_narray = task_params[ts.TaskParams.task_states_narray]
    #task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
    #task_state_action_narray = task_params[ts.TaskParams.task_state_action_narray]
    #feature_matrix = task_params[ts.TaskParams.feature_matrix]
    #print "Start states:"
    #print task_start_state_set
    #print "State space: ", task_states_narray.shape, task_states_narray.size
    #print "Action space: ", task_state_action_narray.shape, task_state_action_narray.size
    #print "Feature matrix: ", feature_matrix.shape, feature_matrix.size
    #_, task_start_states, task_state_action_map, _ = ts.load_state_data()
    #expert_visited_states, expert_state_action_map, _, _ = ts.read_task_data()
    #print "Total number of actions by agents using expert policy is %d" % sf.run_simulation(get_common_policy(task_state_action_map, expert_state_action_map), get_common_policy(task_state_action_map, expert_state_action_map), random.choice(tuple(task_start_states)))

if __name__=='__main__':
    main()
