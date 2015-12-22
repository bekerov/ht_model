#!/usr/bin/env python

from collections import namedtuple

import cPickle as pickle

MAX_BOXES = 8
MAX_BOXES_ACC = MAX_BOXES/2

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

def load_states(states_file_name):
    with open(states_file_name, "rb") as states_file:
        possible_states = pickle.load(states_file)
        possible_start_states = pickle.load(states_file)
        possible_actions = pickle.load(states_file)
    return possible_states, possible_start_states, possible_actions

def state_print(state):
    s = ['State Explanation:\n']
    s.append('\tNumber of robot\'s boxes: ' + str(state.n_r) + '\n')
    s.append('\tNumber of teammate\'s boxes: ' + str(state.n_h) + '\n')
    s.append('\tIs robot transferring? : ' + str(bool(state.t_r)) + '\n')
    s.append('\tIs teammate transferring? : ' + str(bool(state.t_h)) + '\n')
    s.append('\tIs robot holding its box? : ' + str(bool(state.b_r)) + '\n')
    s.append('\tIs robot holding teammate\'s box? : ' + str(bool(state.b_h)) + '\n')
    s.append('\tHas the task completed? ' + str(bool(state.e)) + '\n')
    return ''.join(s)

