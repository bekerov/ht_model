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

expert_data_files_dir = "../data/sample"
task_parameters_file = "../data/task_parameters.pickle"

State = namedtuple("State",
        [   'n_r', # number of robot's boxes on its side 0..MAX_BOXES_ACC
            'n_h', # number of teammate's boxes on its side 0..MAX_BOXES_ACC
            't_r', # Is the robot transferring? Y/N
            't_h', # Is the teammate transferring? Y/N
            'b_r', # number of robot's boxes held by the robot 0..2
            'b_h', # number of teammate's boxes held by the robot 0..2
            'e'    # Is the task completed? Y/N
            ]
        )
n_state_vars = 7

# dictionary which maps index to an action key to a tuple containing an index and explanation for the action
task_actions_expl = {
        'TR': (0, 'Take robot\'s box from table'),
        'K' : (1, 'Keep box on table'),
        'R' : (2, 'Receive box from teammate'),
        'TH': (3, 'Take teammate\'s box from table'),
        'G' : (4, 'Give box to teammate'),
        'WG': (5, 'Wait for teammate to receive'),
        'WS': (6, 'Wait for state change'),
        'X' : (7, 'Exit task'),
        'KB': (8, 'Keep teammate box back at source')
        }
# dictionary which maps index to action key
task_actions_index = { 0: 'TR', 1: 'K', 2: 'R', 3: 'TH', 4: 'G', 5: 'WG', 6: 'WS', 7: 'X', 8: 'KB' }
n_action_vars = len(task_actions_expl)

# A class to hold task param indices to index into task params list after reading
class TaskParams:
    task_states_list = 0
    task_start_state_set = 1
    task_state_action_dict = 2
    feature_matrix = 3
    expert_visited_states_set = 4
    expert_state_action_dict = 5
    expert_feature_expectation = 6
    n_experiments = 7
    time_per_step = 8

def is_valid_task_state(task_state_tup):
    """Function to check if current task is valid, if not it will be pruned
    """
    if task_state_tup.e == 1:
        # task is done, so other elements of the task_state_tup vector cannot be non-zero
        return all(v == 0 for v in task_state_tup[0:-1])
    if (task_state_tup.n_r + task_state_tup.n_h) > MAX_BOXES_ACC:
        # total number of boxes on robot's side must be less than or equal to max number of boxes
        # accessable or zero when robot's side is sorted and robot is waiting for teammate
        return False
    if task_state_tup.t_r == 1 and task_state_tup.b_h == 0:
        # if the robot is transferring a box, then it must be holding at least one of
        # teammate's box for the task_state_tup to be valid
        return False
    if task_state_tup.b_r + task_state_tup.b_h > 2:
        # the robot cannot hold more than two boxes
        return False
    if (task_state_tup.n_r + task_state_tup.n_h) == MAX_BOXES_ACC and task_state_tup.b_h > 0:
        # if robot has all its accessible boxes then,
        # if the robot gets a box, the box was received from a teammate transfer
        # thus box in hand cannot be teammate's box
        return False

    return True

def generate_task_state_set():
    """Function to generate all valid task states (and possible start task states) for the box color sort task
    """
    # list of possible values for each of the state variable
    options = [
            [v for v in range(MAX_BOXES_ACC+1)],
            [v for v in range(MAX_BOXES_ACC+1)],
            [0, 1],
            [0, 1],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1]
            ]

    # generate the task_states and possible_task_states
    task_states_list = list()
    task_start_states_set = list()
    for vector in product(*options):
        task_state_tup = State(*vector)
        if is_valid_task_state(task_state_tup):
            # only add if task_state_tup is valid
            task_states_list.append(task_state_tup)

            if task_state_tup.n_r != MAX_BOXES_ACC and (task_state_tup.n_r + task_state_tup.n_h) == MAX_BOXES_ACC and all (v == 0 for v in task_state_tup[2:]):
                # check if the task_state_tup is a start_state, if it is add it to list
                task_start_states_set.append(task_state_tup)

    logging.info("Total number states (after pruning) for box color sort task: %d", len(task_states_list))
    logging.info("Total number of possible start states: %d", len(task_start_states_set))

    return task_states_list, task_start_states_set

def get_valid_actions(task_state_tup):
    """Function to determine possible actions that can be taken in the given task state
    """
    actions_list = list()
    if all(v == 0 for v in task_state_tup):
        # robot is done with its part and can only wait for teammate to change
        # task_state_tup
        actions_list.append('WS')
    if task_state_tup.n_r and ((task_state_tup.b_r + task_state_tup.b_h) < 2):
        # if there are robot's boxes on the robots side and the robot has a free hand, take it
        actions_list.append('TR')
    if task_state_tup.n_h and ((task_state_tup.b_r + task_state_tup.b_h) < 2) and task_state_tup.t_r == 0:
        # if there are human's boxes on the robots side and the robot has a free hand, take it
        actions_list.append('TH')
    if task_state_tup.b_h > 0 and task_state_tup.t_r == 1:
        # if the robot is transferring it can wait for the teammate to receive
        actions_list.append('WG')
    if task_state_tup.t_h == 1 and ((task_state_tup.b_h + task_state_tup.b_r) < 2):
        # if the teammate is transferring then the robot can receive,
        # provided one of its hands is free
        actions_list.append('R')
    if task_state_tup.b_r > 0:
        # if the robot is holding its box, it can keep
        actions_list.append('K')
    if task_state_tup.b_h > 0 and task_state_tup.t_r == 0:
        # if the robot is holding teammate's box, it can give
        actions_list.append('G')
    if task_state_tup.e == 1:
        # if task is done, robot can exit
        actions_list.append('X')
    if task_state_tup.b_h == 2 and task_state_tup.t_h == 1 and task_state_tup.t_r == 1:
        # if robot is holding teammate's boxes on both hands and teammate is transferring
        # allow the robot to place the book back at source in order to receive the transferred box
        actions_list.append('KB')

    return actions_list

def generate_task_state_action_dict(task_states_list):
    """Function to generate the task state action dict for the box color sort task
    """
    task_state_action_dict = dict([(e, None) for e in task_states_list])
    for task_state_tup in task_states_list:
        task_state_action_dict[task_state_tup] = get_valid_actions(task_state_tup)

    return task_state_action_dict

def get_feature_vector(task_state_tup, current_action):
    """ Function to compute the feature vector given the current task state vector and current action.
    """
    task_state_list = list(task_state_tup)
    state_feature = [1 if task_state_list[0] else 0] + [1 if task_state_list[1] else 0] + task_state_list[2:]
    action_feature = [1 if action == current_action else 0 for action in task_actions_expl.keys()]
    feature_vector = state_feature + action_feature

    return np.array(feature_vector)

def generate_feature_matrix(task_states_list):
    """ Function to generate the feature matrix, that includes features matching the state
        and actions.
    """
    feature_matrix = np.random.rand((len(task_states_list) * n_action_vars), (n_state_vars + n_action_vars))
    for state_idx, task_state_tup in enumerate(task_states_list):
        for action_idx in task_actions_index:
            task_action = task_actions_index[action_idx]
            feature_vector = get_feature_vector(task_state_tup, task_action)
            feature_matrix[state_idx * n_action_vars + action_idx] = feature_vector
    return feature_matrix

def load_experiment_data(task_states_list, task_start_state_set):
    """Function to read exert data files (after manual video processed) and extract visited states, taken actions and feature expectation
    """
    expert_visited_states_set = set()
    expert_state_action_dict = dict()
    total_time_steps = 0 # cumulative number of time steps of all experiments
    total_time = 0 # cumulative time taken in seconds by all experiments
    n_files_read = 0
    expert_feature_expectation = np.zeros(n_state_vars + n_action_vars)

    for expert_file_name in glob.glob(os.path.join(expert_data_files_dir, '*.txt')):
        n_files_read = n_files_read + 1
        with open(expert_file_name, 'r') as expert_file:
            experiment_name = expert_file.readline()[:-1]
            n_time_steps = int(expert_file.readline()) # number of time steps of current experiment
            total_time_steps = total_time_steps + n_time_steps

            for time_step in range(n_time_steps):
                line = expert_file.readline()
                fields = line.split()
                experiment_time = int(fields[0]) # current time t from t_0
                current_action = fields[-1]

                if current_action not in task_actions_expl:
                    logging.error("Filename: %s", expert_file_name)
                    logging.error("Line: %d", time_step+3)
                    logging.error("Action %s not recognized", current_action)
                    sys.exit()

                task_state_tup = State(*map(int, fields[1:-1]))

                if time_step == 0: # check for valid start state
                    if task_state_tup not in task_start_state_set:
                        logging.error("expert_file_name: %s", expert_file_name)
                        logging.error("Line: %d", time_step+3)
                        logging.error("State: \n%s", task_state_print(task_state_tup))
                        logging.error("Not valid start state!")
                        sys.exit()
                else:
                    if task_state_tup not in task_states_list:
                        logging.error("expert_file_name: %s", expert_file_name)
                        logging.error("Line: %d", time_step+3)
                        logging.error("State: %s", task_state_print(task_state_tup))
                        logging.error("Not valid state!")
                        sys.exit()

                expert_feature_expectation = expert_feature_expectation + get_feature_vector(task_state_tup, current_action)
                expert_visited_states_set.add(task_state_tup)
                if task_state_tup not in expert_state_action_dict:
                    expert_state_action_dict[task_state_tup] = dict()
                if current_action in expert_state_action_dict[task_state_tup]:
                    expert_state_action_dict[task_state_tup][current_action] = expert_state_action_dict[task_state_tup][current_action] + 1
                else:
                    expert_state_action_dict[task_state_tup][current_action] = 1
        total_time = total_time + experiment_time/2.0 # dividing by 2.0 since, all the videos were stretched twice for manual processing

    time_per_step = total_time / total_time_steps
    expert_feature_expectation = expert_feature_expectation/n_files_read
    logging.info("Total files read: %d", n_files_read)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    logging.info("mu_e = %s", pformat(expert_feature_expectation))
    logging.info("Total number of expert visited states: %d", len(expert_visited_states_set))

    logging.info("Seconds per time step: %f", round(time_per_step, 2))

    return expert_visited_states_set, expert_state_action_dict, expert_feature_expectation, n_files_read, time_per_step

def write_task_parameters():
    """Function to generate task parameters (states set, state_action dict, feature matrix), convert non numpy structs to numpy and dump to the task_parameters pickle file
    """
    task_states_list, task_start_state_set = generate_task_state_set()
    task_state_action_dict = generate_task_state_action_dict(task_states_list)

    feature_matrix = generate_feature_matrix(task_states_list)
    expert_visited_states_set, expert_state_action_dict, expert_feature_expectation, n_files_read, time_per_step = load_experiment_data(task_states_list, task_start_state_set)

    task_params = [task_states_list, task_start_state_set, task_state_action_dict, feature_matrix, expert_visited_states_set, expert_state_action_dict, expert_feature_expectation, n_files_read, time_per_step]

    with open(task_parameters_file, "wb") as params_file:
        pickle.dump(task_params, params_file)

def load_task_parameters():
    """Function to load the task parameters (state, state_action map, feature matrix)
       from saved disk file
    """
    if not os.path.isfile(task_parameters_file):
        logging.info("Generating task parameters file %s" % task_parameters_file)
        write_task_parameters()

    with open(task_parameters_file, "rb") as params_file:
        task_params = pickle.load(params_file)

    return task_params

def task_state_print(task_state):
    """Function to pretty print the task_state of the task with elaborate explanations
    Arg:
        task_state: task_state of the task
    Returns:
        string: Complete explanation of the task_state returned as a string which can be printed
    """
    s = ['State Explanation:\n']
    s.append('\tNumber of robot\'s boxes: ' + str(task_state.n_r) + '\n')
    s.append('\tNumber of teammate\'s boxes: ' + str(task_state.n_h) + '\n')
    s.append('\tIs robot transferring? : ' + str(bool(task_state.t_r)) + '\n')
    s.append('\tIs teammate transferring? : ' + str(bool(task_state.t_h)) + '\n')
    s.append('\tIs robot holding its box? : ' + str(bool(task_state.b_r)) + '\n')
    s.append('\tIs robot holding teammate\'s box? : ' + str(bool(task_state.b_h)) + '\n')
    s.append('\tHas the task completed? ' + str(bool(task_state.e)) + '\n')
    return ''.join(s)

