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
    task_states, task_start_states, task_state_action_map = ts.load_states()
    _, _, mu_e = ts.read_task_data()
    #simulate_random_poliy(task_start_states, task_state_action_map)
    pi_r1 = init_random_policy(task_state_action_map)
    pi_r2 = init_random_policy(task_state_action_map)

    # we are ignoring the exit task bit (doesn't add any information, always 1 on mu_e) so nstates-1
    mu_r1 = [0] * (ts.n_states - 1)
    mu_r2 = [0] * (ts.n_states - 1)
    nactions_r1 = 0
    nactions_r2 = 0
    trials = 28
    for i in range(trials):
        start_state = random.choice(tuple(task_start_states))
        state_r1 = start_state
        state_r2 = start_state
        while True:
            action_r1 = sf.softmax_select_action(pi_r1[state_r1])
            nactions_r1 = nactions_r1 + 1
            nactions_r2 = nactions_r2 + 1
            action_r2 = sf.softmax_select_action(pi_r2[state_r2])
            if action_r1 == 'X' and action_r2 == 'X':
                break
            state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
            state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
            mu_r1 = [sum(x) for x in zip(mu_r1, ts.get_phi(list(state_r1)))]
            mu_r2 = [sum(x) for x in zip(mu_r2, ts.get_phi(list(state_r2)))]
    mu_r1 = [float(x)/trials for x in mu_r1]
    mu_r2 = [float(x)/trials for x in mu_r2]
    mu_r1_diff = [abs(x - y) for x, y in zip(mu_e, mu_r1)]
    mu_r2_diff = [abs(x - y) for x, y in zip(mu_e, mu_r2)]
    # print "Start state: ", start_state
    # print "nactions r1 = ", nactions_r1
    # print "nactions r2 = ", nactions_r2
    print "mu_e = ", [round(e, 3) for e in mu_e]
    print "mu_r1 ", [round(e, 3) for e in mu_r1]
    print "mu_r2 ", [round(e, 3) for e in mu_r2]
    print "mu_r1_diff = ", [round(e, 3) for e in mu_r1_diff]
    print "mu_r2_diff = ", [round(e, 3) for e in mu_r2_diff]
    print "t_r1 = ", np.linalg.norm(np.array(mu_r1_diff))
    print "t_r2 = ", np.linalg.norm(np.array(mu_r2_diff))
    i = 1
