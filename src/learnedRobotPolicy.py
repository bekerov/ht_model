#!/usr/bin/env python

import sys
import time

from random import choice

import taskSetup as ts
from readData import read_data

def simulate_next_state(state, r_action, h_action):
    return state

if __name__=='__main__':
    possible_states, possible_start_states, possible_actions = ts.load_states(ts.states_file_path)
    state = choice(tuple(possible_states))
    actions = possible_actions[state]
    print state
    print ts.state_print(state)
    print "***********************************************"
    print "Allowed Actions:"
    for i, action in enumerate(actions):
        print "\t", i+1, ts.permitted_actions[action]

