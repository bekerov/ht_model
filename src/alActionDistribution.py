#!/usr/bin/env python

import logging
import random
import pprint
import numpy as np

import taskSetup as ts
import simulationFunctions as sf

# set logging level
logging.basicConfig(level=logging.WARN, format='%(asctime)s-%(levelname)s: %(message)s')

# load task params from pickle file
task_params = ts.load_task_parameters()
task_states_list = task_params[ts.TaskParams.task_states_list]
task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]
feature_matrix = task_params[ts.TaskParams.feature_matrix]
expert_visited_states_set = task_params[ts.TaskParams.expert_visited_states_set]
expert_state_action_dict = task_params[ts.TaskParams.expert_state_action_dict]
n_episodes = task_params[ts.TaskParams.n_episodes]
time_per_step = task_params[ts.TaskParams.time_per_step]

# number of states in task
n_states = len(task_states_list)

# get numpy matrix of task list
state_space = np.zeros((n_states, ts.n_state_vars))
for state_idx, task_state_tup in enumerate(task_states_list):
    state_space[state_idx] = np.array(task_state_tup)

# get numpy value matrix of state action space
# 0 for valid actions and -inf for invalid actions from given state
state_action_space = np.zeros((n_states, ts.n_action_vars))
state_action_space[:] = -np.inf
for state_idx, task_state_tup in enumerate(task_states_list):
    action_idx = [ts.task_actions_dict[_][0] for _ in task_state_action_dict[task_state_tup]]
    np.put(state_action_space[state_idx], action_idx, 0)

# Q-Learning parameters
alpha = 0.2
gamma = 1.0

def compute_random_state_action_distribution():
    """Function to compute a random state action distribution
    """
    random_state_action_dist = np.random.rand(n_states, ts.n_action_vars)
    random_state_action_dist[state_action_space == -np.inf] = 0
    random_state_action_dist = random_state_action_dist / random_state_action_dist.sum(axis=1).reshape((len(random_state_action_dist), 1)) # normalize each row

    return random_state_action_dist

def extract_policy(q_state_action_matrix):
    """Function to extract the policy from q_value matrix
    """
    policy = dict()
    action_keys = {v[0]: k for k, v in ts.task_actions_dict.items()}
    for state_idx, task_state_tup in enumerate(task_states_list):
        action_idx = np.argmax(q_state_action_matrix[state_idx])
        policy[task_state_tup] = action_keys[action_idx]

    return policy

def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    policy = extract_policy(compute_random_state_action_distribution())

    pprint.pprint(policy)

if __name__=='__main__':
    main()
