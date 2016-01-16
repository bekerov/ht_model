#!/usr/bin/env python

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

