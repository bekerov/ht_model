#!/usr/bin/env python

import logging
import random

import taskSetup as ts
import simulationFunctions as sf

"""This module creates an random action distribution for the box color sort task.
"""
# set logging level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')

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

def compute_random_state_action_distribution_dict():
    """Function to compute a random distribution for actions for each task state
    """
    random_state_action_distribution_dict = dict()
    for task_state_tup in task_states_dict.values():
        random_actions = dict()
        actions_dict = task_state_action_dict[task_state_tup]
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
    print "Total number of actions by agents using random policy is %d" % sf.run_simulation(r1_state_action_distribution_dict, r2_state_action_distribution_dict, random.choice(tuple(task_start_state_set)))

if __name__=='__main__':
    simulate_random_state_action_distribution()