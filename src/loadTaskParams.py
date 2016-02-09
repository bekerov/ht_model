#!/usr/bin/env python

import numpy as np
import pprint

import taskSetup as ts

# load task params from pickle file
task_params = ts.load_task_parameters()
task_states_list = task_params[ts.TaskParams.task_states_list]
task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]
feature_matrix = task_params[ts.TaskParams.feature_matrix]
expert_visited_states_set = task_params[ts.TaskParams.expert_visited_states_set]
expert_state_action_dict = task_params[ts.TaskParams.expert_state_action_dict]
expert_feature_expectation = task_params[ts.TaskParams.expert_feature_expectation]
n_experiments = task_params[ts.TaskParams.n_experiments]
time_per_step = task_params[ts.TaskParams.time_per_step]

# number of states in task
n_states = len(task_states_list)

# get numpy matrix of task list
state_space = np.zeros((n_states, ts.n_state_vars))
for state_idx, task_state_tup in enumerate(task_states_list):
    state_space[state_idx] = np.array(task_state_tup)

# get numpy value matrix of state action space
# 0 for valid actions and -inf for invalid actions from given state
state_action_space = np.random.rand(n_states, ts.n_action_vars)
state_action_space[:] = -np.inf
for state_idx, task_state_tup in enumerate(task_states_list):
    action_idx = [ts.task_actions_expl[_][0] for _ in task_state_action_dict[task_state_tup]]
    np.put(state_action_space[state_idx], action_idx, 0)

