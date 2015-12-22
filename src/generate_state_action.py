#!/usr/bin/env python

from collections import namedtuple
from pprint import pprint
from itertools import product
from random import choice

permitted_actions = {
        'WG': 'Wait for teammate to receive',
        'WS': 'Wait for state change',
        'TR': 'Take robot\'s box from table',
        'TH': 'Take teammate\'s box from table',
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

def generate_states():
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

def state_exp(state):
    s = ['State Explanation:\n']
    s.append('\tNumber of robot\'s boxes: ' + str(state.n_r) + '\n')
    s.append('\tNumber of teammate\'s boxes: ' + str(state.n_h) + '\n')
    s.append('\tIs robot transferring? : ' + str(bool(state.t_r)) + '\n')
    s.append('\tIs teammate transferring? : ' + str(bool(state.t_h)) + '\n')
    s.append('\tIs robot holding its box? : ' + str(bool(state.b_r)) + '\n')
    s.append('\tIs robot holding teammate\'s box? : ' + str(bool(state.b_h)) + '\n')
    s.append('\tHas the task completed? ' + str(bool(state.e)) + '\n')
    return ''.join(s)

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

def generate_state_action(states):
    def get_permitted_actions(state):
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
        state_action[state] = get_permitted_actions(state)

    return state_action

def simulate_next_state(state, r_action, h_action):
    return state

if __name__=='__main__':
    states, start_states = generate_states()
    state_action = generate_state_action(states)
    state = choice(tuple(states))
    actions = state_action[state]
    print state
    print state_exp(state)
    print "***********************************************"
    print "Allowed Actions:"
    for i, action in enumerate(actions):
        print "\t", i+1, permitted_actions[action]

