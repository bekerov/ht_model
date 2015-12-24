#!/usr/bin/env python

import sys
import time

from random import choice
from termcolor import colored

from taskSetup import *
from readData import read_data

"""This module creates a common policy for the box color sort task
   based on the most frequent actions taken by humans given a particular
   state
"""

def get_common_policy(state_action):
    policy = dict()
    for state, actions in state_action.items():
        policy[state] = max(actions, key=actions.get)
    return policy

if __name__=='__main__':
    if len(sys.argv) != 2:
        print "Usage: " + sys.argv[0] + " path to data files"
        sys.exit()
    visited_states, taken_actions, time_per_step = read_data(sys.argv[1], "../data/states.pickle")
    policy = get_common_policy(taken_actions)
    print "Total number of visited states: ", len(visited_states)
    print "Seconds per time step: ", round(time_per_step, 2)
    while True:
        state = choice(tuple(visited_states))
        print state
        print state_print(state)
        print colored("Robot\'s action: %s" % permitted_actions[policy[state]], 'green')
        print "**********************************************************"
        user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        if user_input.upper() == 'Q':
            break;
        print "**********************************************************"
        #if policy[state] == 'X':
            #break
        #time.sleep(time_per_step)
