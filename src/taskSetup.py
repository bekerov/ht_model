#!/usr/bin/env python

import os
import sys
import glob
import time

from collections import namedtuple
from itertools import product

import cPickle as pickle

"""This module defines the framework for the box color sort task
   and misc functions that are unique to the task
"""

MAX_BOXES = 8
MAX_BOXES_ACC = MAX_BOXES/2
data_files_path = "../data/sample"
states_file_path = "../data/states.pickle"

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

def read_data(path, states_file_name):
    """Function to read the data files that contains the trajectories of human-human teaming for box color sort task.
    Arg:
        arg1: Path to directory containing the files
    Returns:
        set: all states visited
        dict: dict of dicts mapping states to actions to frequency of that action
        float: time per time step in seconds
    """
    possible_states, possible_start_states, possible_actions = load_states(states_file_name)
    state_action = dict()
    visited_states = set()
    total_steps = 0 # cumulative number of time steps of all experiments
    total_time = 0 # cumulative time taken in seconds by all experiments
    n_files = 0

    for filename in glob.glob(os.path.join(path, '*.txt')):
        n_files = n_files + 1
        with open(filename, 'r') as in_file:
            e_name = in_file.readline()[:-1] # experiment name
            n_steps = int(in_file.readline()) # number of time steps of current experiment
            total_steps = total_steps + n_steps

            for i in range(n_steps):
                line = in_file.readline()
                fields = line.split()
                e_time = int(fields[0]) # time taken in seconds by current experiment
                action = fields[-1]

                if action not in permitted_actions:
                    print "Filename: ", e_name
                    print "Line: ", i+3
                    print "Action %s not recognized" % action
                    sys.exit()

                state_vector = map(int, fields[1:-1])
                state = State(*state_vector)

                if i == 0:
                    if state not in possible_start_states:
                        print "Filename: ", e_name
                        print "Line: ", i+3
                        print "State: ", state
                        print "Not valid start state!"
                        sys.exit()
                else:
                    if state not in possible_states:
                        print "Filename: ", e_name
                        print "Line: ", i+3
                        print "State: ", state
                        print "Not valid state!"
                        sys.exit()

                visited_states.add(state)
                if state not in state_action:
                    state_action[state] = dict()
                if action in state_action[state]:
                    state_action[state][action] = state_action[state][action] + 1
                else:
                    state_action[state][action] = 1
        total_time = total_time + e_time/2.0 # dividing by 2.0 since, all the videos were stretched twice for manual processing

    time_per_step = total_time / total_steps
    print "Total files read: ", n_files
    return visited_states, state_action, time_per_step

def generate_states():
    """Function to generate all valid states for the box color sort task
    Arg:
        None
    Returns:
        frozenset: all valid states
        frozenset: all valid start states
    """
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
    """Function to generate all valid actions for the given states for the box color sort task
    Arg:
        frozenset: set of states for the task
    Returns:
        dict: dict mapping each state to possible actions that can be taken in that state
    """
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

def load_states(states_file_name):
    """Function to load the state framework from saved disk file
    Arg:
        string: state file name
    Returns:
        frozenset: possible states for the task
        frozenset: possible start states for the task
        dict: dict of states mapped to actions available in that state
    """
    if not os.path.isfile(states_file_name):
        print "Generating %s file" % states_file_name
        write_states(states_file_name)
    with open(states_file_name, "rb") as states_file:
        possible_states = pickle.load(states_file)
        possible_start_states = pickle.load(states_file)
        possible_actions = pickle.load(states_file)
    return possible_states, possible_start_states, possible_actions

def write_states(states_file_name):
    """Function to save the state framework to disk as pickle file
    Arg:
        string: state file name
    Returns:
        None
    """
    possible_states, possible_start_states = generate_states()
    possible_actions = generate_actions(possible_states)
    with open(states_file_name, "wb") as states_file:
        pickle.dump(possible_states, states_file)
        pickle.dump(possible_start_states, states_file)
        pickle.dump(possible_actions, states_file)

def state_print(state):
    """Function to pretty print the state of the task with elaborate explanations
    Arg:
        State: state of the task
    Returns:
        string: Complete explanation of the state returned as a string which can be printed
    """
    s = ['State Explanation:\n']
    s.append('\tNumber of robot\'s boxes: ' + str(state.n_r) + '\n')
    s.append('\tNumber of teammate\'s boxes: ' + str(state.n_h) + '\n')
    s.append('\tIs robot transferring? : ' + str(bool(state.t_r)) + '\n')
    s.append('\tIs teammate transferring? : ' + str(bool(state.t_h)) + '\n')
    s.append('\tIs robot holding its box? : ' + str(bool(state.b_r)) + '\n')
    s.append('\tIs robot holding teammate\'s box? : ' + str(bool(state.b_h)) + '\n')
    s.append('\tHas the task completed? ' + str(bool(state.e)) + '\n')
    return ''.join(s)

