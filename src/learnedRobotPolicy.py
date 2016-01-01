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
    print "Total number of actions by agents using random policy is %d" % sf.run_simulation(init_random_policy(task_state_action_map), init_random_policy(task_state_action_map), random.choice(tuple(task_start_states)))
