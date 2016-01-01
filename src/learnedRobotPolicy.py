#!/usr/bin/env python

import sys
import time
import cPickle as pickle

import random
from termcolor import colored

import taskSetup as ts
import simulationFunctions as sf

if __name__=='__main__':
    task_sates, task_start_states, task_state_action_map = ts.load_states()
    pi_1 = sf.init_random_policy(task_state_action_map)
    pi_2 = sf.init_random_policy(task_state_action_map)
    state_r1 = random.choice(tuple(task_start_states))
    state_r2 = state_r1
    nactions = 0
    while True:
        nactions = nactions + 1
        action_r1 = sf.softmax_select_action(pi_1[state_r1])
        action_r2 = sf.softmax_select_action(pi_2[state_r2])
        print colored("state_r1 before: %s" % str(state_r1), 'red')
        print colored("action_r1: %s" % ts.task_actions[action_r1], 'red')
        print colored("state_r2 before: %s" % str(state_r2), 'cyan')
        print colored("action_r2: %s" % ts.task_actions[action_r2], 'cyan')
        if action_r1 == 'X' or action_r2 == 'X':
            break
        state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
        state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
        print colored("state_r1 after: %s" % str(state_r1), 'red')
        print colored("state_r2 after: %s" % str(state_r2), 'cyan')
        print "******************************************************************************"
        print "******************************************************************************"

    print "Total number of actions by agents using random policy is %d" % nactions
