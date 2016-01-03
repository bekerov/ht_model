#!/usr/bin/env python

import sys
import time
import logging
import cPickle as pickle
import random
import numpy as np

from termcolor import colored

import taskSetup as ts
import simulationFunctions as sf

def init_random_policy(task_state_action_map):
    random_policy = dict()

    for task_state, state_actions in task_state_action_map.items():
        actions = dict()
        for action in state_actions:
            actions[action] = random.random()
        actions = {k: float(v)/ total for total in (sum(actions.values()),) for k, v in actions.items()}
        random_policy[task_state] = actions
    return random_policy

def simulate_random_poliy(task_start_states, task_state_action_map):
    print "Total number of actions by agents using random policy is %d" % sf.run_simulation(init_random_policy(task_state_action_map), init_random_policy(task_state_action_map), random.choice(tuple(task_start_states)))

def get_feature_vector(state_vector, phi):
    return [sum(x) for x in zip(phi, [1 if state_vector[0] else 0] + [1 if state_vector[0] else 0] + state_vector[2:])]


if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')
    task_states, task_start_states, task_state_action_map = ts.load_states()
    pi_r1 = init_random_policy(task_state_action_map)
    pi_r2 = init_random_policy(task_state_action_map)
    phi_r1 = [0] * 6
    phi_r2 = [0] * 6
    total_phi_r1 = [0] * 6
    total_phi_r2 = [0] * 6
    start_state = random.choice(tuple(task_start_states))
    state_r1 = start_state
    state_r2 = start_state
    while True:
        action_r1 = sf.softmax_select_action(pi_r1[state_r1])
        action_r2 = sf.softmax_select_action(pi_r2[state_r2])
        if action_r1 == 'X' and action_r2 == 'X':
            break
        state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
        state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
        phi_r1 = get_feature_vector(list(state_r1), phi_r1)
        phi_r2 = get_feature_vector(list(state_r2), phi_r2)
    total_phi_r1 = [sum(x) for x in zip(total_phi_r1, phi_r1)]
    total_phi_r2 = [sum(x) for x in zip(total_phi_r2, phi_r2)]
    mu_r1 = [round(x/float(total_phi_r1[-1]), 3) for x in total_phi_r1]
    mu_r2 = [round(x/float(total_phi_r2[-1]), 3) for x in total_phi_r2]
    print mu_r1
    print mu_r2

    #n_iterations = 10
    #for i in range(n_iterations):
        #phi_r1 = [0] * 6
        #phi_r2 = [0] * 6
        #start_state = random.choice(tuple(task_start_states))
        #random_policy = init_random_policy(task_state_action_map)
        #state_r1 = start_state
        #state_r2 = start_state
        #while True:
            #action_r1 = sf.softmax_select_action(random_policy[state_r1])
            #action_r2 = sf.softmax_select_action(random_policy[state_r2])
            #if action_r1 == 'X' and action_r2 == 'X':
                #break
            #state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
            #state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
            #phi_r1 = get_feature_vector(list(state_r1), phi_r1)
            #phi_r2 = get_feature_vector(list(state_r2), phi_r2)
        #total_phi_r1 = [sum(x) for x in zip(total_phi_r1, phi_r1)]
        #total_phi_r2 = [sum(x) for x in zip(total_phi_r2, phi_r2)]

    #print phi_r1
    #print phi_r2
    #print total_phi_r1
    #print total_phi_r2
    #simulate_random_poliy(task_start_states, task_state_action_map)
