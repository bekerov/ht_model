#!/usr/bin/env python

import sys
import time

from random import choice
from random import random

import taskSetup as ts

def init_random_policy(possible_state_action):
    policy = dict()

    for possible_state, possible_actions in possible_state_action.items():
        actions = dict()
        sumation = 0
        for possible_action in possible_actions:
            actions[possible_action] = random()
            sumation = sumation + actions[possible_action]
        for possible_action in possible_actions:
            actions[possible_action] = actions[possible_action] / sumation
        policy[possible_state] = actions
    return policy

def simulate_next_state(current_state, r_action, h_action):
    return current_state

if __name__=='__main__':
    possible_states, possible_start_states, possible_state_actions = ts.load_states(ts.states_file_path)
    state = choice(tuple(possible_states))
    actions = possible_state_actions[state]
    pi_1 = init_random_policy(possible_state_actions)
    pi_2 = init_random_policy(possible_state_actions)
    # print state
    # print ts.state_print(state)
    # print "***********************************************"
    # print "Allowed Actions:"
    # for i, action in enumerate(actions):
        # print "\t", i+1, ts.permitted_actions[action]

