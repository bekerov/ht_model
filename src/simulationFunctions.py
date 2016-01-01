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

def init_random_policy(task_state_action_map):
    random_policy = dict()

    for task_state, state_actions in task_state_action_map.items():
        actions = dict()
        for action in state_actions:
            actions[action] = random.random()
        actions = {k: float(v)/ total for total in (sum(actions.values()),) for k, v in actions.items()}
        random_policy[task_state] = actions
    return random_policy

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
