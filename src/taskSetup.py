#!/usr/bin/env python

import os
import sys
import glob
import logging
import numpy as np

from pprint import pformat
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
task_data_path = "../data/task_data.pickle"

task_actions = {
        'WG': 'Wait for teammate to receive',
        'WS': 'Wait for state change',
        'TR': 'Take robot\'s box from table',
        'TH': 'Take teammate\'s box from table',
        'G': 'Give box to teammate',
        'K': 'Keep box on table',
        'R': 'Receive box from teammate',
        'X': 'Exit task'
        }

n_states = 7
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
        if state.n_r != MAX_BOXES_ACC and (state.n_r + state.n_h) == MAX_BOXES_ACC and all (v == 0 for v in state[2:]):
            start_states.add(state)

    logging.info("Total number states (after pruning) for box color sort task: %d", len(states))
    return states, start_states

def generate_actions(states):
    """Function to generate all valid actions for the given states for the box color sort task
    Arg:
        frozenset: set of states for the task
    Returns:
        dict: dict mapping each state to possible actions that can be taken in that state
    """
    def generate_task_actions(state):
        actions = set()
        if all(v == 0 for v in state):
            # robot is done with its part and can only wait for teammate to change
            # state
            actions.add('WS')
        if state.n_r and state.b_r == 0:
            # if there are robot's boxes on the robots side, take it
            actions.add('TR')
        if state.n_h and state.b_h == 0:
            # if there are human's boxes on the robots side, take it
            actions.add('TH')
        if state.b_h == 1 and state.t_r == 1:
            # if the robot is transferring it can wait for the teammate to receive
            actions.add('WG')
        if state.t_h == 1 and ((state.b_h + state.b_r) < 2):
            # if the teammate is transferring then the robot can receive,
            # provided one of its hands is free
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
        state_action[state] = generate_task_actions(state)

    return state_action

def get_phi(task_state_vector):
    return np.array([1 if task_state_vector[0] else 0] + [1 if task_state_vector[1] else 0] + task_state_vector[2:-1])

def write_task_data():
    """Function to read the data files that contains the trajectories of human-human teaming for box color sort task and write out the processed python data structions.
    """
    task_states, task_start_states, task_state_action_map = load_states()
    expert_state_action_map = dict()
    expert_visited_states = set()
    total_steps = 0 # cumulative number of time steps of all experiments
    total_time = 0 # cumulative time taken in seconds by all experiments
    n_files = 0
    sum_phi = [0] * (n_states - 1) # we are ignoring the exit task bit (doesn't add any information, always 1 on mu_e)

    for filename in glob.glob(os.path.join(data_files_path, '*.txt')):
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

                if action not in task_actions:
                    logging.error("Filename: %s", e_name)
                    logging.error("Line: %d", i+3)
                    logging.error("Action %s not recognized", action)
                    sys.exit()

                task_state_vector = map(int, fields[1:-1])
                sum_phi = [sum(x) for x in zip(sum_phi, get_phi(task_state_vector))]
                task_state = State(*task_state_vector)

                if i == 0:
                    if task_state not in task_start_states:
                        logging.error("Filename: %s", e_name)
                        logging.error("Line: %d", i+3)
                        logging.error("State: %s", str(task_state))
                        logging.error("Not valid start state!")
                        sys.exit()
                else:
                    if task_state not in task_states:
                        logging.error("Filename: %s", e_name)
                        logging.error("Line: %d", i+3)
                        logging.error("State: %s", str(task_state))
                        logging.error("Not valid start state!")
                        sys.exit()

                expert_visited_states.add(task_state)
                if task_state not in expert_state_action_map:
                    expert_state_action_map[task_state] = dict()
                if action in expert_state_action_map[task_state]:
                    expert_state_action_map[task_state][action] = expert_state_action_map[task_state][action] + 1
                else:
                    expert_state_action_map[task_state][action] = 1
        total_time = total_time + e_time/2.0 # dividing by 2.0 since, all the videos were stretched twice for manual processing

    time_per_step = total_time / total_steps
    mu_e = [float(x)/n_files for x in sum_phi]
    logging.info("Generating %s file" % task_data_path)
    with open(task_data_path, "wb") as task_data_file:
        pickle.dump(expert_visited_states, task_data_file)
        pickle.dump(expert_state_action_map, task_data_file)
        pickle.dump(mu_e, task_data_file)
        pickle.dump(time_per_step, task_data_file)
        pickle.dump(n_files, task_data_file)

def read_task_data():
    """Function to read the data files that contains the trajectories of human-human teaming for box color sort task.
    Arg:
        None
    Returns:
       set: expert_visited_states
       dict: dict of expert visited states mapped to action which is mapped to its frequency
       mu_e: feature expection for apprenticeship learning
       time_per_step: number of seconds per time step for realistic simulation
    """
    if not os.path.isfile(task_data_path):
        write_task_data()
    with open(task_data_path, "rb") as task_data_file:
        expert_visited_states = pickle.load(task_data_file)
        expert_state_action_map = pickle.load(task_data_file)
        mu_e = pickle.load(task_data_file)
        time_per_step = pickle.load(task_data_file)
        n_files = pickle.load(task_data_file)
    logging.info("Total files read: %d", n_files)
    logging.info("mu_e = %s", pformat(mu_e))
    logging.info("Total number of expert visited states: %d", len(expert_visited_states))
    logging.info("Seconds per time step: %f", round(time_per_step, 2))
    return expert_visited_states, expert_state_action_map, np.array(mu_e), n_files

def load_states():
    """Function to load the state framework from saved disk file
    Arg:
        None
    Returns:
        frozenset: possible states for the task
        frozenset: possible start states for the task
        dict: dict of states mapped to actions available in that state
    """
    if not os.path.isfile(states_file_path):
        write_states()
    with open(states_file_path, "rb") as states_file:
        task_states = pickle.load(states_file)
        task_start_states = pickle.load(states_file)
        task_state_action_map = pickle.load(states_file)
    return task_states, task_start_states, task_state_action_map

def write_states():
    """Function to save the state framework to disk as pickle file
    """
    task_states, task_start_states = generate_states()
    task_state_action_map = generate_actions(task_states)
    logging.info("Generating %s file" % states_file_path)
    with open(states_file_path, "wb") as states_file:
        pickle.dump(task_states, states_file)
        pickle.dump(task_start_states, states_file)
        pickle.dump(task_state_action_map, states_file)

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

