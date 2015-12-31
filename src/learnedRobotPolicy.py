#!/usr/bin/env python

import sys
import time

from random import choice
from random import random
from termcolor import colored

import cPickle as pickle

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
        actions = {k: float(v)/ total for total in (sum(actions.values()),) for k, v in actions.items()}
        policy[possible_state] = actions
    return policy

def simulate_next_state(action, state_r1, state_r2):
    state_r1_prime = state_r1
    state_r2_prime = state_r2
    # if action == 'WG' or action == 'WS':
    if action == 'TR':
        state_r1_prime = ts.State(state_r1.n_r-1, state_r1.n_h, state_r1.t_r, state_r1.t_h, 1, state_r1.b_h, state_r1.e)
    if action == 'TH':
        state_r1_prime = ts.State(state_r1.n_r, state_r1.n_h-1, state_r1.t_r, state_r1.t_h, state_r1.b_r, 1, state_r1.e)
    if action == 'K':
        state_r1_prime = ts.State(state_r1.n_r, state_r1.n_h, state_r1.t_r, state_r1.t_h, 0, state_r1.b_h, state_r1.e)
    if action == 'G':
        state_r1_prime = ts.State(state_r1.n_r, state_r1.n_h, 1, state_r1.t_h, state_r1.b_r, state_r1.b_h, state_r1.e)
        state_r2_prime = ts.State(state_r2.n_r, state_r2.n_h, state_r2.t_r, 1, state_r2.b_r, state_r2.b_h, state_r2.e)
    if action == 'R':
        state_r1_prime = ts.State(state_r1.n_r, state_r1.n_h, state_r1.t_r, 0, 1, state_r1.b_h, state_r1.e)
        state_r2_prime = ts.State(state_r2.n_r, state_r2.n_h, 0, state_r2.t_h, state_r2.b_r, 0, state_r2.e)
    if (sum(list(state_r1)) + sum(list(state_r2))) == 0:
        state_r1_prime = ts.State(state_r1.n_r, state_r1.n_h, state_r1.t_r, state_r1.t_h, state_r1.b_r, state_r1.b_h, 1)
        state_r2_prime = ts.State(state_r2.n_r, state_r2.n_h, state_r2.t_r, state_r2.t_h, state_r2.b_r, state_r2.b_h, 1)

    return state_r1_prime, state_r2_prime

if __name__=='__main__':
    possible_states, possible_start_states, possible_state_actions = ts.load_states(ts.states_file_path)
    pi_1 = init_random_policy(possible_state_actions)
    pi_2 = init_random_policy(possible_state_actions)
    # verify_action_selection(pi_1, state)
    state_r1 = choice(tuple(possible_start_states))
    state_r2 = state_r1
    with open("policy.pickle", "rb") as p_file:
        policy = pickle.load(p_file)
    # print state
    while True:
        # action_r1 = softmax_select_action(pi_1[state_r1])
        # action_r2 = softmax_select_action(pi_2[state_r2])
        if state_r1 not in policy:
            action_r1 = choice(tuple(possible_state_actions[state_r1]))
        else:
            action_r1 = softmax_select_action(policy[state_r1])
        if state_r2 not in policy:
            action_r2 = choice(tuple(possible_state_actions[state_r2]))
        else:
            action_r2 = softmax_select_action(policy[state_r2])

        print colored("state_r1 before: %s" % str(state_r1), 'red')
        print colored("action_r1: %s" % ts.permitted_actions[action_r1], 'red')
        print colored("state_r2 before: %s" % str(state_r2), 'blue')
        print colored("action_r2: %s" % ts.permitted_actions[action_r2], 'blue')
        if action_r1 == 'X' or action_r2 == 'X':
            break
        state_r1, state_r2 = simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
        state_r2, state_r1 = simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
        print colored("state_r1 after: %s" % str(state_r1), 'red')
        print colored("state_r2 after: %s" % str(state_r2), 'blue')
        print "******************************************************************************"
        print "******************************************************************************"
