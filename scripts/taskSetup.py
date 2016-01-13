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

###################### Change the path ###################
expert_data_files_dir = "../data/sample"
task_parameters_file = "data/task_parameters.pickle"
#states_file_path = "../data/states.pickle"
#task_data_path = "../data/task_data.pickle"

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
n_state_vars = 7

# dictionary which maps action key to a tuple containing an index and explanation for the action
task_actions = {
        'TR': (0, 'Take robot\'s box from table'),
        'K' : (1, 'Keep box on table'),
        'R' : (2, 'Receive box from teammate'),
        'TH': (3, 'Take teammate\'s box from table'),
        'G' : (4, 'Give box to teammate'),
        'WG': (5, 'Wait for teammate to receive'),
        'WS': (6, 'Wait for state change'),
        'X' : (7, 'Exit task')
        }
n_action_vars = len(task_actions)

# A class to hold task param indices to index into task params list after reading
class TaskParams:
    task_states = 0
    possible_task_start_states = 1
    task_state_action_map = 2
    feature_matrix = 3

def is_valid_task_state(task_state):
    """Function to check if current task is valid, if not it will be pruned
    """
    if task_state.e == 1:
        # task is done, so other elements of the task_state vector cannot be non-zero
        return all(v == 0 for v in task_state[0:-1])
    if (task_state.n_r + task_state.n_h) > MAX_BOXES_ACC:
        # total number of boxes on robot's side must be less than or equal to max number of boxes
        # accessable or zero when robot's side is sorted and robot is waiting for teammate
        return False
    if task_state.t_r == 1 and task_state.b_h != 1:
        # if the robot is transferring a box, then it must be holding the
        # teammate's box for the task_state to be valid
        return False
    if (task_state.n_r + task_state.n_h) == MAX_BOXES_ACC and task_state.b_h == 1:
        # if robot has all its accessible boxes then,
        # if the robot gets a box, the box was received from a teammate transfer
        # thus box in hand cannot be teammate's box
        return False

    return True

def generate_task_state_space():
    """Function to generate all valid task states (and possible start task states) for the box color sort task
    """
    # list of possible values for each of the state variable
    options = [
            [v for v in range(MAX_BOXES_ACC+1)],
            [v for v in range(MAX_BOXES_ACC+1)],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1]
            ]

    # generate the task_states and possible_task_states
    task_states = np.empty((0, n_state_vars))
    possible_task_start_states = list()
    for vector in product(*options):
        task_state = State(*vector)
        if is_valid_task_state(task_state):
            # only add if task_state is valid
            task_states = np.vstack((task_states, task_state))

            if task_state.n_r != MAX_BOXES_ACC and (task_state.n_r + task_state.n_h) == MAX_BOXES_ACC and all (v == 0 for v in task_state[2:]):
                # check if the task_state is a start_state, if it is add it to list
                possible_task_start_states.append(task_state)

    logging.info("Total number states (after pruning) for box color sort task: %d", len(task_states))
    logging.info("Total number of possible start states: %d", len(possible_task_start_states))

    return task_states, possible_task_start_states

def get_valid_actions(task_state):
    """Function to determine possible actions that can be taken in the given task_state
    """
    actions = list()
    if all(v == 0 for v in task_state):
        # robot is done with its part and can only wait for teammate to change
        # task_state
        actions.append('WS')
    if task_state.n_r and task_state.b_r == 0:
        # if there are robot's boxes on the robots side, take it
        actions.append('TR')
    if task_state.n_h and task_state.b_h == 0:
        # if there are human's boxes on the robots side, take it
        actions.append('TH')
    if task_state.b_h == 1 and task_state.t_r == 1:
        # if the robot is transferring it can wait for the teammate to receive
        actions.append('WG')
    if task_state.t_h == 1 and ((task_state.b_h + task_state.b_r) < 2):
        # if the teammate is transferring then the robot can receive,
        # provided one of its hands is free
        actions.append('R')
    if task_state.b_r == 1:
        # if the robot is holding its box, it can keep
        actions.append('K')
    if task_state.b_h == 1 and task_state.t_r == 0:
        # if the robot is holding teammate's box, it can give
        actions.append('G')
    if task_state.e == 1:
        # if task is done, robot can exit
        actions.append('X')

    return actions

def generate_task_state_action_map(task_states):
    """Function to generate the task state action map for the box color sort task
    """
    # generate the task_state_action_map which is a matrix SxA having value one for those action's indices which are
    # valid for that task_state
    task_state_action_map = np.empty((0, n_action_vars))
    for task_state in task_states:
        # get the indices of the valid actions for the task_state from task_actions
        action_idx = [task_actions[_][0] for _ in get_valid_actions(State(*task_state.tolist()))]

        # create a row for the current task_state
        current_task_state_map = np.zeros(n_action_vars)
        np.put(current_task_state_map, action_idx, 1)

        # add the row to the matrix
        task_state_action_map = np.vstack((task_state_action_map, current_task_state_map))

    return task_state_action_map

def get_feature_vector(task_state_vector, current_action):
    """ Function to compute the feature vector given the current task state vector and current action.
    """
    state_feature = [1 if task_state_vector[0] else 0] + [1 if task_state_vector[1] else 0] + task_state_vector[2:]
    action_feature = [1 if action == current_action else 0 for action in task_actions.keys()]
    feature_vector = state_feature + action_feature

    return np.array(feature_vector)

def generate_feature_matrix(task_states):
    """ Function to generate the feature matrix, that includes features matching the state
        and actions.
    """
    feature_matrix = np.empty((0, (n_state_vars + n_action_vars)))
    for task_state in task_states:
        for task_action in task_actions:
            feature_vector = get_feature_vector(list(State(*task_state.tolist())), task_action)
            feature_matrix = np.vstack((feature_matrix, feature_vector))

    return feature_matrix

def write_task_parameters():
    """Function to generate task parameters (state, state_action map, feature matrix)
       and dump to the task_parameters pickle file
    """
    task_states, possible_task_start_states = generate_task_state_space()
    task_state_action_map = generate_task_state_action_map(task_states)
    feature_matrix = generate_feature_matrix(task_states)

    with open(task_parameters_file, "wb") as params_file:
        pickle.dump(task_states, params_file)
        pickle.dump(possible_task_start_states, params_file)
        pickle.dump(task_state_action_map, params_file)
        pickle.dump(feature_matrix, params_file)

def load_task_parameters():
    """Function to load the task parameters (state, state_action map, feature matrix)
       from saved disk file
    """
    if not os.path.isfile(task_parameters_file):
        logging.info("Generating task parameters file %s" % task_parameters_file)
        write_task_parameters()

    with open(task_parameters_file, "rb") as params_file:
        task_states = pickle.load(params_file)
        possible_task_start_states = pickle.load(params_file)
        task_state_action_map = pickle.load(params_file)
        feature_matrix = pickle.load(params_file)

    task_params = [task_states, possible_task_start_states, task_state_action_map, feature_matrix]
    return task_params

def load_experiment_data():
    task_params = load_task_parameters()
    task_states = task_params[TaskParams.task_states]
    possible_task_start_states = task_params[TaskParams.possible_task_start_states]
    task_state_action_map = task_params[TaskParams.task_state_action_map]
    expert_visited_states = set()
    expert_state_action_map = dict()
    total_time_steps = 0 # cumulative number of time steps of all experiments
    total_time = 0 # cumulative time taken in seconds by all experiments
    n_files_read = 0
    expert_feature_expectation = np.zeros(n_state_vars + n_action_vars)

    for filename in glob.glob(os.path.join(expert_data_files_dir, '*.txt')):
        n_files_read = n_files_read + 1
        with open(filename, 'r') as expert_file:
            experiment_name = expert_file.readline()[:-1]
            n_time_steps = int(expert_file.readline()) # number of time steps of current experiment
            total_time_steps = total_time_steps + n_time_steps
            print "Filename: ", experiment_name
            print "Time steps: ", n_time_steps

    print "Number of files read: ", n_files_read
    print "Total number of time steps: ", total_time_steps


#def write_task_data():
    #"""Function to read the data files that contains the trajectories of human-human teaming for box color sort task and write out the processed python data structions.
    #"""
    #task_states, task_start_states, task_state_action_map, _ = load_state_data()
    #task_states = set(task_states.values())
    #expert_state_action_map = dict()
    #expert_visited_states = set()
    #total_steps = 0 # cumulative number of time steps of all experiments
    #total_time = 0 # cumulative time taken in seconds by all experiments
    #n_files = 0
    ##mu_e = np.zeros(n_state_vars-1) # we are ignoring the exit task bit (doesn't add any information, always 1 on mu_e)
    #mu_e = np.zeros(n_state_vars + len(task_actions))

    #for filename in glob.glob(os.path.join(data_files_path, '*.txt')):
        #n_files = n_files + 1
        #with open(filename, 'r') as in_file:
            #e_name = in_file.readline()[:-1] # experiment name
            #n_steps = int(in_file.readline()) # number of time steps of current experiment
            #total_steps = total_steps + n_steps

            #for i in range(n_steps):
                #line = in_file.readline()
                #fields = line.split()
                #e_time = int(fields[0]) # time taken in seconds by current experiment
                #current_action = fields[-1]

                #if current_action not in task_actions:
                    #logging.error("Filename: %s", e_name)
                    #logging.error("Line: %d", i+3)
                    #logging.error("current_action %s not recognized", current_action)
                    #sys.exit()

                #task_state_vector = map(int, fields[1:-1])
                #mu_e = mu_e + get_phi(task_state_vector, current_action)
                #task_state = State(*task_state_vector)

                #if i == 0:
                    #if task_state not in task_start_states:
                        #logging.error("Filename: %s", e_name)
                        #logging.error("Line: %d", i+3)
                        #logging.error("State: %s", str(task_state))
                        #logging.error("Not valid start state!")
                        #sys.exit()
                #else:
                    #if task_state not in task_states:
                        #logging.error("Filename: %s", e_name)
                        #logging.error("Line: %d", i+3)
                        #logging.error("State: %s", str(task_state))
                        #logging.error("Not valid start state!")
                        #sys.exit()

                #expert_visited_states.add(task_state)
                #if task_state not in expert_state_action_map:
                    #expert_state_action_map[task_state] = dict()
                #if current_action in expert_state_action_map[task_state]:
                    #expert_state_action_map[task_state][current_action] = expert_state_action_map[task_state][current_action] + 1
                #else:
                    #expert_state_action_map[task_state][current_action] = 1
        #total_time = total_time + e_time/2.0 # dividing by 2.0 since, all the videos were stretched twice for manual processing

    #time_per_step = total_time / total_steps
    #mu_e = mu_e/n_files
    ##mu_e = mu_e/np.linalg.norm(mu_e)
    #logging.info("Generating %s file" % task_data_path)
    #with open(task_data_path, "wb") as task_data_file:
        #pickle.dump(expert_visited_states, task_data_file)
        #pickle.dump(expert_state_action_map, task_data_file)
        #pickle.dump(mu_e, task_data_file)
        #pickle.dump(time_per_step, task_data_file)
        #pickle.dump(n_files, task_data_file)

#def read_task_data():
    #"""Function to read the data files that contains the trajectories of human-human teaming for box color sort task.
    #Arg:
        #None
    #Returns:
       #set: expert_visited_states
       #dict: dict of expert visited states mapped to action which is mapped to its frequency
       #mu_e: feature expection for apprenticeship learning
       #time_per_step: number of seconds per time step for realistic simulation
    #"""
    #if not os.path.isfile(task_data_path):
        #write_task_data()
    #with open(task_data_path, "rb") as task_data_file:
        #expert_visited_states = pickle.load(task_data_file)
        #expert_state_action_map = pickle.load(task_data_file)
        #mu_e = pickle.load(task_data_file)
        #time_per_step = pickle.load(task_data_file)
        #n_files = pickle.load(task_data_file)
    #logging.info("Total files read: %d", n_files)
    #logging.info("mu_e = %s", pformat(mu_e))
    #logging.info("Total number of expert visited states: %d", len(expert_visited_states))
    #logging.info("Seconds per time step: %f", round(time_per_step, 2))
    #return expert_visited_states, expert_state_action_map, mu_e, n_files

#def load_state_data():
    #"""Function to load the state framework from saved disk file
    #Arg:
        #None
    #Returns:
        #frozenset: possible states for the task
        #frozenset: possible start states for the task
        #dict: dict of states mapped to actions available in that state
    #"""
    #if not os.path.isfile(states_file_path):
        #write_state_data()
    #with open(states_file_path, "rb") as states_file:
        #task_states = pickle.load(states_file)
        #task_start_states = pickle.load(states_file)
        #task_state_action_map = pickle.load(states_file)
        #phi = pickle.load(states_file)
    #return task_states, task_start_states, task_state_action_map, phi

#def write_state_data():
    #"""Function to save the state framework to disk as pickle file
    #"""
    #task_states, task_start_states = generate_states()
    #task_state_action_map = generate_actions(task_states.values())
    #phi = generate_phi(task_states.values())
    #logging.info("Generating %s file" % states_file_path)
    #with open(states_file_path, "wb") as states_file:
        #pickle.dump(task_states, states_file)
        #pickle.dump(task_start_states, states_file)
        #pickle.dump(task_state_action_map, states_file)
        #pickle.dump(phi, states_file)


def task_state_print(task_state):
    """Function to pretty print the task_state of the task with elaborate explanations
    Arg:
        task_state: task_state of the task
    Returns:
        string: Complete explanation of the task_state returned as a string which can be printed
    """
    s = ['task_state Explanation:\n']
    s.append('\tNumber of robot\'s boxes: ' + str(task_state.n_r) + '\n')
    s.append('\tNumber of teammate\'s boxes: ' + str(task_state.n_h) + '\n')
    s.append('\tIs robot transferring? : ' + str(bool(task_state.t_r)) + '\n')
    s.append('\tIs teammate transferring? : ' + str(bool(task_state.t_h)) + '\n')
    s.append('\tIs robot holding its box? : ' + str(bool(task_state.b_r)) + '\n')
    s.append('\tIs robot holding teammate\'s box? : ' + str(bool(task_state.b_h)) + '\n')
    s.append('\tHas the task completed? ' + str(bool(task_state.e)) + '\n')
    return ''.join(s)

