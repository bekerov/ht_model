#!/usr/bin/env python

from collections import namedtuple
from pprint import pprint
from random import choice
from itertools import product

possible_actions = {
                        'WG': 'Wait for teammate to receive',
                        'WS': 'Wait for state change',
                        'T': 'Take box from table',
                        'G': 'Give box to teammate',
                        'K': 'Keep box on table',
                        'R': 'Receive box from teammate',
                        'X': 'Exit task'
                   }

MAX_BOXES = 8
MAX_BOXES_ACC = MAX_BOXES/2
State = namedtuple("State",
                        [   'n_r', # number of robot's boxes on its side 0..MAX_BOXES_ACC
                            'n_h', # number of teammate's boxes on its side 0..MAX_BOXES_ACC
                            't_r', # Is the robot transferring? Y/N
                            't_h', # Is the teammate transferring? Y/N
                            'b_r', # Is the robot holding its box? Y/N
                            'b_h', # Is the robot holding the teammate's box? Y/N
                            'e'    # Is the task completed? Y/N
                        ]
                  )

def get_start_states():
    start_states = set()
    for i in range(MAX_BOXES_ACC+1):
        start_states.add(State(i, MAX_BOXES_ACC-i, 0, 0, 0, 0, 0))
    return start_states

def generate_states():
    states = list()
    state_vals = [
                    [v for v in range(MAX_BOXES_ACC+1)],
                    [v for v in range(MAX_BOXES_ACC+1)],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1]
                 ]
    for s in product(*state_vals):
        state = State(*s)
        if is_valid_state(state):
            states.append(state)

    return frozenset(states)

def is_valid_state(state):
    if state.e == 1:
        # task is done, so other elements of the state vector cannot be non-zero
        return all(v == 0 for v in state[0:-1])
    if state.n_r + state.n_h > MAX_BOXES_ACC:
        # total number of boxes on robot's side must be less than or equal to max number of books
        # or zero when robot's side is sorted and robot is waiting for teammate
        return False
    if state.t_r == 1 and state.b_h != 1:
        # if the robot is transferring a box, then it must be holding the
        # teammate's box for the state to be valid
        return False
    if state.t_h == 1 and state.b_r == 1 and state.b_h == 1:
        # if robot is holding teammate's box and teammate is transferring, the
        # robot will not be holding its own box as well
        return False
    return True

def get_valid_actions(state):
    actions = list()
    if all(v == 0 for v in state):
        # robot is done with its part and can only wait for teammate to change
        # state
        actions.append('WS')
    return actions

if __name__=='__main__':
    start_states = get_start_states()
    #state = choice(tuple(start_states))
    #state = State(0, 0, 0, 0, 0, 0, 0)
    #print is_valid_state(state)
    #print state
    states = generate_states()
    state_action = dict([ (elem, None) for elem in states ])
    for state in state_action:
        state_action[state] = get_valid_actions(state)
    pprint(state_action)

