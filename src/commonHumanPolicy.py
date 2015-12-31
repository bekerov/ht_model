#!/usr/bin/env python

import sys
import time

from random import choice
from random import random
from termcolor import colored
from pprint import pprint

import cPickle as pickle

import taskSetup as ts

"""This module creates a common policy for the box color sort task
   based on the most frequent actions taken by humans given a particular
   state
"""

def softmax_select_action(actions):
   r = random()
   upto = 0
   for action, probs in actions.items():
      if upto + probs >= r:
         return action
      upto += probs
   assert False, "Shouldn't get here"


def get_common_policy(state_action):
    """Function to extract the common policy given the state action mapping from human-human task
    Arg:
        dict: dict of dict mapping states to actions which in turn is mapped to its frequency of use
    Returns:
        dict: mapping state to the most frequent action of that particular state
    """
    policy = dict()
    # for state, actions in state_action.items():
        # policy[state] = max(actions, key=actions.get)
    # policy = dict()
    for possible_state, possible_actions in taken_actions.items():
        actions = {k: float(v)/total for total in (sum(possible_actions.values()),) for k, v in possible_actions.items()}
        policy[possible_state] = actions
    return policy

if __name__=='__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else ts.data_files_path
    visited_states, taken_actions, time_per_step = ts.read_data(path, ts.states_file_path)
    policy = get_common_policy(taken_actions)
    with open("policy.pickle", "wb") as p_file:
            pickle.dump(policy, p_file)
    print "Total number of visited states: ", len(visited_states)
    print "Seconds per time step: ", round(time_per_step, 2)
    while True:
        state = choice(tuple(visited_states))
        print state
        print ts.state_print(state)
        print colored("Robot\'s action: %s" % ts.permitted_actions[softmax_select_action(policy[state])], 'green')
        print "**********************************************************"
        user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        if user_input.upper() == 'Q':
            break;
        print "**********************************************************"
