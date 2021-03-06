#!/usr/bin/env python

import sys
import random
import logging

from termcolor import colored

import taskSetup as ts
from helperFuncs import select_random_action
from loadTaskParams import *

"""This module contains the functions for simulating the box color sort task between two agents
"""

logging.basicConfig(format='')
lgr = logging.getLogger("simulationFunctions.py")
lgr.setLevel(level=logging.WARN)

def simulate_next_state(current_action, my_current_state, teammate_current_state):
    my_next_state = my_current_state
    teammate_next_state = teammate_current_state

    # if current_action is waiting (either WS or WG) no state is changed

    if current_action == 'TR':
        # current_action is grab robot's box, so only my_current_state changes
        # decreasing robot's accessible boxes and setting robot holding its box to 1
        my_next_state = ts.State(my_current_state.n_r-1, my_current_state.n_h, my_current_state.t_r, my_current_state.t_h, my_current_state.b_r+1, my_current_state.b_h, my_current_state.e)

    if current_action == 'TH':
        # current_action is grab teammate's box, so only my_current_state changes
        # decreasing teammate's accessible boxes and setting robot holding teammate's box to 2
        my_next_state = ts.State(my_current_state.n_r, my_current_state.n_h-1, my_current_state.t_r, my_current_state.t_h, my_current_state.b_r, my_current_state.b_h+1, my_current_state.e)

    if current_action == 'K':
        # current_action is keep box on table, so only my_current_state changes
        # setting robot holding its box to 0
        my_next_state = ts.State(my_current_state.n_r, my_current_state.n_h, my_current_state.t_r, my_current_state.t_h, my_current_state.b_r-1, my_current_state.b_h, my_current_state.e)

    if current_action == 'G':
        # current_action is give box table, so affects both state
        # robot is transferring
        my_next_state = ts.State(my_current_state.n_r, my_current_state.n_h, 1, my_current_state.t_h, my_current_state.b_r, my_current_state.b_h, my_current_state.e)
        teammate_next_state = ts.State(teammate_current_state.n_r, teammate_current_state.n_h, teammate_current_state.t_r, 1, teammate_current_state.b_r, teammate_current_state.b_h, teammate_current_state.e)
    if current_action == 'R':
        # current_action is receive box table, so affects both state
        # robot is receving
        my_next_state = ts.State(my_current_state.n_r, my_current_state.n_h, my_current_state.t_r, 0, my_current_state.b_r+1, my_current_state.b_h, my_current_state.e)
        teammate_next_state = ts.State(teammate_current_state.n_r, teammate_current_state.n_h, 0, teammate_current_state.t_h, teammate_current_state.b_r, teammate_current_state.b_h-1, teammate_current_state.e)

    # if current_action == 'KB':
        # # if current action is keep human box back at source, increment human box count and
        # # make one hand free
        # my_next_state = ts.State(my_current_state.n_r, my_current_state.n_h+1, my_current_state.t_r, my_current_state.t_h, my_current_state.b_r, my_current_state.b_h-1, my_current_state.e)

    if (sum(list(my_current_state)) + sum(list(teammate_current_state))) == 0:
        # if all the states are zero for both, then both are waiting
        # in this case, task is done, so set end bit to 1
        my_next_state = ts.State(my_current_state.n_r, my_current_state.n_h, my_current_state.t_r, my_current_state.t_h, my_current_state.b_r, my_current_state.b_h, 1)
        teammate_next_state = ts.State(teammate_current_state.n_r, teammate_current_state.n_h, teammate_current_state.t_r, teammate_current_state.t_h, teammate_current_state.b_r, teammate_current_state.b_h, 1)

    return my_next_state, teammate_next_state

def run_simulation(r1_dist, r2_dist, start_state):
    """
        Function to run the simulation based on the start state and the agent's distributions
    """
    r1_state_tup = start_state
    r2_state_tup = start_state
    n_actions = 0
    while True:
        n_actions = n_actions + 1
        r1_action = select_random_action(r1_dist[task_states_list.index(r1_state_tup)])
        r2_action = select_random_action(r2_dist[task_states_list.index(r2_state_tup)])
        r1_p = r1_state_tup
        r2_p = r2_state_tup

        lgr.info("%s", colored("Agent 1 state: %s" % str(r1_p), 'red', attrs = ['bold']))
        lgr.info("%s\n", colored("Agent 2 state: %s" % str(r2_p), 'cyan', attrs = ['bold']))

        lgr.info("%s", colored("Agent 1 action: %s" % ts.task_actions_expl[r1_action][1], 'red', attrs = ['bold']))
        lgr.info("%s\n", colored("Agent 2 action: %s" % ts.task_actions_expl[r2_action][1], 'cyan', attrs = ['bold']))

        if r1_action == 'X' and r2_action == 'X':
            break

        # states are flipped because from each agent's perspective it is the robot and
        # the other agent is the human
        r1_state_tup, r2_state_tup = simulate_next_state(r1_action, r1_state_tup, r2_state_tup) # first agent acting
        r2_state_tup, r1_state_tup = simulate_next_state(r2_action, r2_state_tup, r1_state_tup) # second agent acting
        if r1_action == 'R' or r1_action == 'G' or r2_action == 'R' or r2_action == 'G':
            lgr.info("%s", colored("Agent 1 state after both agent's actions  : %s" % str(r1_state_tup), 'red', attrs = ['bold']))
            lgr.info("%s\n", colored("Agent 2 state after both agent's actions  : %s" % str(r2_state_tup), 'cyan', attrs = ['bold']))

        lgr.info("%s", colored("************************************************************************************************************", 'white', attrs = ['bold']))
        lgr.info("%s", colored("************************************************************************************************************", 'white', attrs = ['bold']))
        if lgr.getEffectiveLevel() == logging.DEBUG:
            user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
            if user_input.upper() == 'Q':
                break
    return n_actions

