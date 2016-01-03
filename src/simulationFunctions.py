#!/usr/bin/env python

import random
import logging

from termcolor import colored

import taskSetup as ts

"""This module contains the functions for simulating the box color sort task between two agents
"""

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


def simulate_next_state(action, my_state, teammate_state):
    my_next_state = my_state
    teammate_next_state = teammate_state

    # if action is waiting (either WS or WG) no state is changed

    if action == 'TR':
        # Action is grab robot's box, so only my_state changes
        # decreasing robot's accessible boxes and setting robot holding its box to 1
        my_next_state = ts.State(my_state.n_r-1, my_state.n_h, my_state.t_r, my_state.t_h, 1, my_state.b_h, my_state.e)
    if action == 'TH':
        # Action is grab teammate's box, so only my_state changes
        # decreasing teammate's accessible boxes and setting robot holding teammate's box to 1
        my_next_state = ts.State(my_state.n_r, my_state.n_h-1, my_state.t_r, my_state.t_h, my_state.b_r, 1, my_state.e)
    if action == 'K':
        # Action is keep box on table, so only my_state changes
        # setting robot holding its box to 0
        my_next_state = ts.State(my_state.n_r, my_state.n_h, my_state.t_r, my_state.t_h, 0, my_state.b_h, my_state.e)
    if action == 'G':
        # Action is give box table, so affects both state
        # robot is transferring
        my_next_state = ts.State(my_state.n_r, my_state.n_h, 1, my_state.t_h, my_state.b_r, my_state.b_h, my_state.e)
        teammate_next_state = ts.State(teammate_state.n_r, teammate_state.n_h, teammate_state.t_r, 1, teammate_state.b_r, teammate_state.b_h, teammate_state.e)
    if action == 'R':
        # Action is receive box table, so affects both state
        # robot is receving
        my_next_state = ts.State(my_state.n_r, my_state.n_h, my_state.t_r, 0, 1, my_state.b_h, my_state.e)
        teammate_next_state = ts.State(teammate_state.n_r, teammate_state.n_h, 0, teammate_state.t_h, teammate_state.b_r, 0, teammate_state.e)
    if (sum(list(my_state)) + sum(list(teammate_state))) == 0:
        # if all the states are zero for both, then both are waiting
        # in this case, task is done, so set end bit to 1
        my_next_state = ts.State(my_state.n_r, my_state.n_h, my_state.t_r, my_state.t_h, my_state.b_r, my_state.b_h, 1)
        teammate_next_state = ts.State(teammate_state.n_r, teammate_state.n_h, teammate_state.t_r, teammate_state.t_h, teammate_state.b_r, teammate_state.b_h, 1)

    return my_next_state, teammate_next_state

def run_simulation(pi_1, pi_2, start_state):
    state_r1 = start_state
    state_r2 = start_state
    n_actions = 0
    while True:
        n_actions = n_actions + 1
        action_r1 = softmax_select_action(pi_1[state_r1])
        action_r2 = softmax_select_action(pi_2[state_r2])
        logging.debug("%s", colored("state_r1 before: %s" % str(state_r1), 'red'))
        logging.debug("%s", colored("action_r1: %s" % ts.task_actions[action_r1], 'red'))
        logging.debug("%s", colored("state_r2 before: %s" % str(state_r2), 'cyan'))
        logging.debug("%s", colored("action_r2: %s" % ts.task_actions[action_r2], 'cyan'))

        if action_r1 == 'X' and action_r2 == 'X':
            break

        # states are flipped because from each agent's perspective it is the robot and
        # the other agent is the human
        state_r1, state_r2 = simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
        state_r2, state_r1 = simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
        logging.debug("%s", colored("state_r1 after: %s" % str(state_r1), 'red'))
        logging.debug("%s", colored("state_r2 after: %s" % str(state_r2), 'cyan'))
        logging.debug("******************************************************************************")
        logging.debug("******************************************************************************")

    return n_actions
