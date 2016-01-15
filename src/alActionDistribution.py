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
state_action_space = np.zeros((n_states, ts.n_action_vars))
state_action_space[:] = -np.inf
for state_idx, task_state_tup in enumerate(task_states_list):
    action_idx = [ts.task_actions_expl[_][0] for _ in task_state_action_dict[task_state_tup]]
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
    """Function to extract the policy from q_value matrix using argmax
    """
    policy = dict()
    for state_idx, task_state_tup in enumerate(task_states_list):
        action_idx = np.argmax(q_state_action_matrix[state_idx])
        policy[task_state_tup] = ts.task_actions_index[action_idx]

    return policy

def select_random_action(state_action_vector):
    """Function to select a random action based on the its probability in the state action distribution
    """
    action_idx = np.random.choice(np.arange(state_action_vector.size), p = state_action_vector)
    action = ts.task_actions_index[action_idx]
    return action

def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)

    r1_state_action_dist = compute_random_state_action_distribution()
    r2_state_action_dist = compute_random_state_action_distribution()
    start_state = random.choice(task_start_state_set)
    r1_state_idx = task_states_list.index(start_state)
    r2_state_idx = r1_state_idx
    r1_state_tup = start_state
    r2_state_tup = start_state
    print feature_matrix.shape, feature_matrix.size
    for state_idx, task_state_tup in enumerate(task_states_list):
        for action_idx in ts.task_actions_index:
            task_action = ts.task_actions_index[action_idx]
            cmpr = feature_matrix[state_idx][action_idx] == ts.get_feature_vector(task_state_tup, task_action)
            if not np.all(cmpr):
                print ts.get_feature_vector(task_state_tup, task_action)
                print feature_matrix[state_idx][action_idx]
                break
    print "Done"
    #while True:
        #r1_action = select_random_action(r1_state_action_dist[r1_state_idx])
        #r2_action = select_random_action(r2_state_action_dist[r2_state_idx])

        #print r1_action
        #print r1_state_tup
        #print r2_action
        #print r2_state_tup

        #if r1_action == 'X' and r2_action == 'X':
            #break

        #r1_state_tup, r2_state_tup = sf.simulate_next_state(r1_action, r1_state_tup, r2_state_tup) # first agent acting
        #r2_state_tup, r1_state_tup = sf.simulate_next_state(r2_action, r2_state_tup, r1_state_tup) # second agent acting

        ## update the state indices for both agents
        #r1_state_idx = task_states_list.index(r1_state_tup)
        #r2_state_idx = task_states_list.index(r2_state_tup)

        #print "**********************************************************"
        #user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        #if user_input.upper() == 'Q':
         #break;
        #print "**********************************************************"

if __name__=='__main__':
    main()
