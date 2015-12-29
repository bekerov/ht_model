#!/usr/bin/env python

import sys
import time

from random import choice
from random import random

import taskSetup as ts

def softmax_select_action(actions):
   r = random()
   upto = 0
   for action, probs in actions.items():
      if upto + probs >= r:
         return action
      upto += probs
   assert False, "Shouldn't get here"

def verify_action_selection(pi, state):
    counts = dict()
    for i in range(10000):
        action = softmax_select_action(pi[state])
        if action not in counts:
            counts[action] = 1
        else:
            counts[action] = counts[action] + 1
    counts = {k: float(v)/total for total in (sum(counts.values()),) for k, v in counts.items()}
    print pi[state]
    print counts


def init_random_policy(possible_state_action):
    policy = dict()

    for possible_state, possible_actions in possible_state_action.items():
        actions = dict()
        for possible_action in possible_actions:
            actions[possible_action] = random()
        actions = {k: v / total for total in (sum(actions.values()),) for k, v in actions.items()}
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
    verify_action_selection(pi_1, state)

    # print state
    # print ts.state_print(state)
    # print "***********************************************"
    # print "Allowed Actions:"
    # for i, action in enumerate(actions):
        # print "\t", i+1, ts.permitted_actions[action]
