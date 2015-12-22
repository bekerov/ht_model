#!/usr/bin/env python

import cPickle as pickle

from collections import namedtuple
from itertools import product

from taskSetup import *

def generate_states():
    def is_valid_state(state):
        if state.e == 1:
            # task is done, so other elements of the state vector cannot be non-zero
            return all(v == 0 for v in state[0:-1])
        if (state.n_r + state.n_h) > MAX_BOXES_ACC:
            # total number of boxes on robot's side must be less than or equal to max number of boxes
            # accessable or zero when robot's side is sorted and robot is waiting for teammate
            return False
        if state.t_r == 1 and state.b_h != 1:
            # if the robot is transferring a box, then it must be holding the
            # teammate's box for the state to be valid
            return False
        if (state.n_r + state.n_h) == MAX_BOXES_ACC and state.b_h == 1:
            # if robot has all its accessible boxes then,
            # if the robot gets a box, the box was received from a teammate transfer
            # thus box in hand cannot be teammate's box
            return False
        return True
    states = set()
    vals = [
            [v for v in range(MAX_BOXES_ACC+1)],
            [v for v in range(MAX_BOXES_ACC+1)],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1]
            ]
    for s in product(*vals):
        state = State(*s)
        if is_valid_state(state):
            states.add(state)

    start_states = set()
    for state in states:
        if (state.n_r + state.n_h) == MAX_BOXES_ACC and all (v == 0 for v in state[2:]):
            start_states.add(state)

    return frozenset(states), frozenset(start_states)

def generate_actions(states):
    def generate_permitted_actions(state):
        actions = set()
        if all(v == 0 for v in state):
            # robot is done with its part and can only wait for teammate to change
            # state
            actions.add('WS')
        if state.n_r:
            # if there are robot's boxes on the robots side, take it
            actions.add('TR')
        if state.n_h:
            # if there are human's boxes on the robots side, take it
            actions.add('TH')
        # if (state.n_r + state.n_h):
            # actions.add('T')
        if state.b_h == 1 and state.t_r == 1:
            # if the robot is transferring it can wait for the teammate to receive
            actions.add('WG')
        if state.t_h == 1:
            # if the teammate is transferring then the robot can receive
            actions.add('R')
        if state.b_r == 1:
            # if the robot is holding its box, it can keep
            actions.add('K')
        if state.b_h == 1 and state.t_r == 0:
            # if the robot is holding teammate's box, it can give
            actions.add('G')
        if state.e == 1:
            # if task is done, robot can exit
            actions.add('X')
        return actions

    state_action = dict([ (elem, None) for elem in states ])
    for state in state_action:
        state_action[state] = generate_permitted_actions(state)

    return state_action

if __name__=='__main__':
    possible_states, possible_start_states = generate_states()
    possible_actions = generate_actions(possible_states)
    with open("states.pickle", "wb") as states_file:
        pickle.dump(possible_states, states_file)
        pickle.dump(possible_start_states, states_file)
        pickle.dump(possible_actions, states_file)
