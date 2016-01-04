#!/usr/bin/env python

import sys
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



if __name__=='__main__':
    logging.basicConfig(level=logging.WARN, format='%(asctime)s-%(levelname)s: %(message)s')
    task_states, task_start_states, task_state_action_map, phi = ts.load_state_data()
    _, _, mu_e, n_trials = ts.read_task_data()
    print phi.shape
    print phi.size
    #simulate_random_poliy(task_start_states, task_state_action_map)
    #pi_r1 = init_random_policy(task_state_action_map)
    #pi_r2 = init_random_policy(task_state_action_map)

    ## we are ignoring the exit task bit (doesn't add any information, always 1 on mu_e) so nstates-1
    #mu = np.zeros(ts.n_states-1)
    #for i in range(n_trials):
        #start_state = random.choice(tuple(task_start_states))
        #state_r1 = start_state
        #state_r2 = start_state
        #while True:
            #action_r1 = sf.softmax_select_action(pi_r1[state_r1])
            #action_r2 = sf.softmax_select_action(pi_r2[state_r2])
            #if action_r1 == 'X' and action_r2 == 'X':
                #break
            #state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
            #state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
            #mu = mu + ts.get_phi(list(state_r1))
    #mu = mu/n_trials
    #w = np.absolute(mu_e - mu)
    #t = np.linalg.norm(w)
    #np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    #print "mu_e = ", mu_e
    #print "mu = ", mu
    #print "w = ", w
    #print "t = ", np.round(t, 3)

