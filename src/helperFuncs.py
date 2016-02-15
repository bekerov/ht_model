#!/usr/bin/env python

import random
import numpy as np

from loadTaskParams import *

def compute_random_state_action_distribution():
    """Function to compute a random state action distribution
    """
    #random_state_action_dist = np.random.rand(n_states, ts.n_action_vars)
    #random_state_action_dist[state_action_space == -np.inf] = 0
    random_state_action_dist = np.ones((n_states, ts.n_action_vars))
    random_state_action_dist[state_action_space == -np.inf] = 0
    #random_state_action_dist = random_state_action_dist * np.random.rand(n_states, ts.n_action_vars)
    random_state_action_dist = random_state_action_dist / random_state_action_dist.sum(axis=1).reshape((len(random_state_action_dist), 1)) # normalize each row

    return random_state_action_dist

def extract_best_policy(state_action_dist):
    """Function to extract the policy from q_value matrix using argmax
    """
    policy = dict()
    for state_idx, task_state_tup in enumerate(task_states_list):
        action_idx = np.argmax(state_action_dist[state_idx])
        policy[task_state_tup] = ts.task_actions_index[action_idx]

    return policy

def extract_state_action_distribution_dict(state_action_dist):
    """Function to convert the numpy state_action distribution to
    dict of dicts for comparing againts expert and random policy
    """
    learned_state_action_distribution_dict = dict()
    for state_idx, task_state_tup in enumerate(task_states_list):
        learned_task_actions = dict()
        for action_idx in ts.task_actions_index:
            if state_action_dist[state_idx][action_idx] != 0:
                learned_task_actions[ts.task_actions_index[action_idx]] = state_action_dist[state_idx][action_idx]

        learned_state_action_distribution_dict[task_state_tup] = learned_task_actions
    return learned_state_action_distribution_dict

def softmax(w, t = 1.0):
    """Function to calculate the softmax of a matrix, normalizing each row for probability
    """
    e = np.exp(w / t)
    dist = e/e.sum(axis=1).reshape((len(e), 1))
    return dist

def select_random_action(state_action_vector):
    """Function to select a random action based on the its probability in the state action distribution
    """
    action_idx = np.random.choice(np.arange(state_action_vector.size), p = state_action_vector)
    action = ts.task_actions_index[action_idx]

    return action
