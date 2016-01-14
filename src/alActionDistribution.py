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

# Q-Learning parameters
alpha = 0.2
gamma = 1.0

def compute_random_state_action_distribution_dict():
    """Function to compute a random distribution for actions for each task state
    """
    random_state_action_distribution_dict = dict()
    for task_state_tup in task_states_set:
        random_actions = dict()
        actions_dict = task_state_action_dict[task_state_tup]
        for action in actions_dict:
            random_actions[action] = random.random()
        random_actions = {k: float(v)/total for total in (sum(random_actions.values()),) for k, v in random_actions.items()}
        random_state_action_distribution_dict[task_state_tup] = random_actions

    return random_state_action_distribution_dict

def simulate_random_state_action_distribution_dict():
    """Function to simulate a random action distribution for both the agents
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')

    r1_state_action_distribution_dict = compute_random_state_action_distribution_dict()
    r2_state_action_distribution_dict = compute_random_state_action_distribution_dict()
    print "Total number of actions by agents using random policy is %d" % sf.run_simulation(r1_state_action_distribution_dict, r2_state_action_distribution_dict, random.choice(tuple(task_start_state_set)))

def main():
    logging.basicConfig(level=logging.WARN, format='%(asctime)s-%(levelname)s: %(message)s')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)

if __name__=='__main__':
    main()
