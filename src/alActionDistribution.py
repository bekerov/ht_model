#!/usr/bin/env python

import logging
import random
import pprint
import numpy as np

import taskSetup as ts
import simulationFunctions as sf

# load task params from pickle file
task_params = ts.load_task_parameters()
task_states_set = task_params[ts.TaskParams.task_states_set]
task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]
feature_matrix = task_params[ts.TaskParams.feature_matrix]
expert_visited_states_set = task_params[ts.TaskParams.expert_visited_states_set]
expert_state_action_dict = task_params[ts.TaskParams.expert_state_action_dict]
n_episodes = task_params[ts.TaskParams.n_episodes]
time_per_step = task_params[ts.TaskParams.time_per_step]

# number of states in task
n_states = len(task_states_set)

# Q-Learning parameters
alpha = 0.2
gamma = 1.0

def main():
    logging.basicConfig(level=logging.WARN, format='%(asctime)s-%(levelname)s: %(message)s')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)

if __name__=='__main__':
    main()
