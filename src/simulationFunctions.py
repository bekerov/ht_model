#!/usr/bin/env python

import time
import random

import taskSetup as ts

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

def softmax_select_action(actions):
   r = random.random()
   upto = 0
   for action, probs in actions.items():
      if upto + probs >= r:
         return action
      upto += probs
   assert False, "Shouldn't get here"

def init_random_policy(possible_state_action):
    policy = dict()

    for possible_state, possible_actions in possible_state_action.items():
        actions = dict()
        for possible_action in possible_actions:
            actions[possible_action] = random.random()
        actions = {k: float(v)/ total for total in (sum(actions.values()),) for k, v in actions.items()}
        policy[possible_state] = actions
    return policy

def simulate_next_state(action, state_r1, state_r2):
    state_r1_prime = state_r1
    state_r2_prime = state_r2
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
