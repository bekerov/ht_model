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
# value is 0 for valid actions and -inf for invalid actions from given state
state_action_space = np.zeros((n_states, ts.n_action_vars))
state_action_space[:] = -np.inf
for state_idx, task_state_tup in enumerate(task_states_list):
    actions = task_state_action_dict[task_state_tup]
    for a in actions:
        action_idx = ts.task_actions_dict[a][0]
        state_action_space[state_idx][action_idx] = 0

# Q-Learning parameters
alpha = 0.2
gamma = 1.0

def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    print task_states_list[10]
    print state_space[10]
    print task_state_action_dict[task_states_list[10]]
    print state_action_space[10]

if __name__=='__main__':
    main()
