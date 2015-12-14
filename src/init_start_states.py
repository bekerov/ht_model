#!/usr/bin/env python

from collections import namedtuple
from pprint import pprint
from random import choice
from itertools import product

MAX_BOXES = 8
MAX_BOXES_ACC = MAX_BOXES/2

State = namedtuple("State", ['n_r', 'n_h', 't_r', 't_h', 'b_r', 'b_h', 'e'])

def get_start_states():
    start_states = set()
    for i in range(MAX_BOXES_ACC+1):
        state = State(n_r = i, n_h = MAX_BOXES_ACC - i, t_r = 0, t_h = 0, b_r = 0, b_h = 0, e = 0)
        start_states.add(state)
    return start_states

def is_valid_state(state):
    if state.e == 1:
        # task is done, so other elements of the state vector cannot be non-zero
        return all(v == 0 for v in state[0:-1])
    if state.n_r + state.n_h > MAX_BOXES_ACC:
        # total number of boxes on robot's side must be less than or equal to max number of books
        # or zero when robot's side is sorted and robot is waiting for human
        return False
    if state.t_r == 1 and state.b_h != 1:
        # if the robot is transferring a box, then it must be holding the
        # human's box for the state to be valid
        return False
    if state.t_h == 1 and state.b_r == 1 and state.b_h == 1:
        # if robot is holding human's box and human is transferring, the
        # robot will not be holding its own box as well
        return False
    return True

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
    for state in product(*state_vals):
        if is_valid_state(State(*state)):
            states.append(state)

    return states

if __name__=='__main__':
    #start_states = get_start_states()
    #state = choice(tuple(start_states))
    #state = State(0, 0, 0, 0, 0, 0, 0)
    #print is_valid_state(state)
    #print state
    states = generate_states()
    print len(states)

