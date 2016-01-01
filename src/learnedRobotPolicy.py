#!/usr/bin/env python

import sys
import time
import logging
import cPickle as pickle
import random

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

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')
    task_sates, task_start_states, task_state_action_map = ts.load_states()
    start_state = random.choice(tuple(task_start_states))
    pi_1 = init_random_policy(task_state_action_map)
    pi_2 = init_random_policy(task_state_action_map)
    nactions = sf.run_simulation(pi_1, pi_2, start_state)
    #state_r1 = random.choice(tuple(task_start_states))
    #state_r2 = state_r1
    #nactions = 0
    #while True:
        #nactions = nactions + 1
        #action_r1 = sf.softmax_select_action(pi_1[state_r1])
        #action_r2 = sf.softmax_select_action(pi_2[state_r2])
        #logging.debug("%s", colored("state_r1 before: %s" % str(state_r1), 'red'))
        #logging.debug("%s", colored("action_r1: %s" % ts.task_actions[action_r1], 'red'))
        #logging.debug("%s", colored("state_r2 before: %s" % str(state_r2), 'cyan'))
        #logging.debug("%s", colored("action_r2: %s" % ts.task_actions[action_r2], 'cyan'))
        #if action_r1 == 'X' or action_r2 == 'X':
            #break

        ## states are flipped because from each agent's perspective it is the robot and
        ## the other agent is the human
        #state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
        #state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
        #logging.debug("%s", colored("state_r1 after: %s" % str(state_r1), 'red'))
        #logging.debug("%s", colored("state_r2 after: %s" % str(state_r2), 'cyan'))
        #logging.debug("******************************************************************************")
        #logging.debug("******************************************************************************")

    print "Total number of actions by agents using random policy is %d" % nactions
