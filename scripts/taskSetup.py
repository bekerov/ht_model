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
task_actions_dict = {
        'TR': (0, 'Take robot\'s box from table'),
        'K' : (1, 'Keep box on table'),
        'R' : (2, 'Receive box from teammate'),
        'TH': (3, 'Take teammate\'s box from table'),
        'G' : (4, 'Give box to teammate'),
        'WG': (5, 'Wait for teammate to receive'),
        'WS': (6, 'Wait for state change'),
        'X' : (7, 'Exit task')
        }
n_action_vars = len(task_actions_dict)

# A class to hold task param indices to index into task params list after reading
class TaskParams:
    task_states_narray = 0
    task_start_state_set = 1
    task_state_action_narray = 2
    feature_matrix = 3

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
    if task_state_tup.t_r == 1 and task_state_tup.b_h != 1:
        # if the robot is transferring a box, then it must be holding the
        # teammate's box for the task_state_tup to be valid
        return False
    if (task_state_tup.n_r + task_state_tup.n_h) == MAX_BOXES_ACC and task_state_tup.b_h == 1:
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
            [0, 1],
            [0, 1],
            [0, 1]
            ]

    # generate the task_states and possible_task_states
    #task_states = np.empty((0, n_state_vars))
    task_states_set = set()
    task_start_states_set = list()
    for vector in product(*options):
        task_state_tup = State(*vector)
        if is_valid_task_state(task_state_tup):
            # only add if task_state_tup is valid
            #task_state_tups = np.vstack((task_state_tups_set, task_state_tup))
            task_states_set.add(task_state_tup)

            if task_state_tup.n_r != MAX_BOXES_ACC and (task_state_tup.n_r + task_state_tup.n_h) == MAX_BOXES_ACC and all (v == 0 for v in task_state_tup[2:]):
                # check if the task_state_tup is a start_state, if it is add it to list
                task_start_states_set.append(task_state_tup)

    logging.info("Total number states (after pruning) for box color sort task: %d", len(task_states_set))
    logging.info("Total number of possible start states: %d", len(task_start_states_set))

    return task_states_set, task_start_states_set

def get_valid_actions(task_state_tup):
    """Function to determine possible actions that can be taken in the given task_state_tup
    """
    actions_list = list()
    if all(v == 0 for v in task_state_tup):
        # robot is done with its part and can only wait for teammate to change
        # task_state_tup
        actions_list.append('WS')
    if task_state_tup.n_r and task_state_tup.b_r == 0:
        # if there are robot's boxes on the robots side, take it
        actions_list.append('TR')
    if task_state_tup.n_h and task_state_tup.b_h == 0:
        # if there are human's boxes on the robots side, take it
        actions_list.append('TH')
    if task_state_tup.b_h == 1 and task_state_tup.t_r == 1:
        # if the robot is transferring it can wait for the teammate to receive
        actions_list.append('WG')
    if task_state_tup.t_h == 1 and ((task_state_tup.b_h + task_state_tup.b_r) < 2):
        # if the teammate is transferring then the robot can receive,
        # provided one of its hands is free
        actions_list.append('R')
    if task_state_tup.b_r == 1:
        # if the robot is holding its box, it can keep
        actions_list.append('K')
    if task_state_tup.b_h == 1 and task_state_tup.t_r == 0:
        # if the robot is holding teammate's box, it can give
        actions_list.append('G')
    if task_state_tup.e == 1:
        # if task is done, robot can exit
        actions_list.append('X')

    return actions_list

def generate_task_state_action_dict(task_states_set):
    """Function to generate the task state action dict for the box color sort task
    """
    # generate the task action dict which is a matrix SxA having value one for those action's indices which are
    # valid for that task_state
    #task_state_action_map = np.empty((0, n_action_vars))
    task_state_action_dict = dict([(e, None) for e in task_states_set])
    for task_state_tup in task_states_set:
        task_state_action_dict[task_state_tup] = get_valid_actions(task_state_tup)
        ## get the indices of the valid actions for the task_state from task_actions_dict
        #action_idx = [task_actions_dict[_][0] for _ in get_valid_actions(State(*task_state.tolist()))]

        ## create a row for the current task_state
        #current_task_state_map = np.zeros(n_action_vars)
        #np.put(current_task_state_map, action_idx, 1)

        ## add the row to the matrix
        #task_state_action_map = np.vstack((task_state_action_map, current_task_state_map))

    return task_state_action_dict

def get_feature_vector(task_state_tup, current_action):
    """ Function to compute the feature vector given the current task state vector and current action.
    """
    state_feature = [1 if task_state_tup[0] else 0] + [1 if task_state_tup[1] else 0] + task_state_tup[2:]
    action_feature = [1 if action == current_action else 0 for action in task_actions_dict.keys()]
    feature_vector = state_feature + action_feature

    return np.array(feature_vector)

def generate_feature_matrix(task_states_set):
    """ Function to generate the feature matrix, that includes features matching the state
        and actions.
    """
    feature_matrix = np.empty((0, (n_state_vars + n_action_vars)))
    for task_state_tup in task_states_set:
        for task_action in task_actions_dict:
            feature_vector = get_feature_vector(list(task_state_tup), task_action)
            feature_matrix = np.vstack((feature_matrix, feature_vector))

    return feature_matrix

def write_task_parameters():
    """Function to generate task parameters (states set, state_action dict, feature matrix), convert non numpy structs to numpy and dump to the task_parameters pickle file
    """
    task_states_set, task_start_state_set = generate_task_state_set()
    task_states_narray = np.empty((0, n_state_vars))
    for task_state_tup in task_states_set:
        task_states_narray = np.vstack((task_states_narray, task_state_tup))

    task_state_action_dict = generate_task_state_action_dict(task_states_set)
    task_state_action_narray = np.empty((0, n_action_vars))
    for task_state_tup, actions_list in task_state_action_dict.items():
        # get the indices of the valid actions for the task_state from task_actions_dict
        action_idx = [task_actions_dict[_][0] for _ in get_valid_actions(task_state_tup)]

        # create a row for the current task_state
        current_task_state_vector = np.zeros(n_action_vars)
        np.put(current_task_state_vector, action_idx, 1)

        # add the row to the matrix
        task_state_action_narray = np.vstack((task_state_action_narray, current_task_state_vector))

    feature_matrix = generate_feature_matrix(task_states_set)

    task_params = [task_states_narray, task_start_state_set, task_state_action_narray, feature_matrix]
    with open(task_parameters_file, "wb") as params_file:
        pickle.dump(task_params, params_file)
        #pickle.dump(task_state_space, params_file)
        #pickle.dump(possible_task_start_states, params_file)
        #pickle.dump(task_state_action_map, params_file)
        #pickle.dump(feature_matrix, params_file)

def load_task_parameters():
    """Function to load the task parameters (state, state_action map, feature matrix)
       from saved disk file
    """
    if not os.path.isfile(task_parameters_file):
        logging.info("Generating task parameters file %s" % task_parameters_file)
        write_task_parameters()

    with open(task_parameters_file, "rb") as params_file:
        task_params = pickle.load(params_file)
        #task_states = pickle.load(params_file)
        #possible_task_start_states = pickle.load(params_file)
        #task_state_action_map = pickle.load(params_file)
        #feature_matrix = pickle.load(params_file)

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

    for expert_file_name in glob.glob(os.path.join(expert_data_files_dir, '*.txt')):
        n_files_read = n_files_read + 1
        with open(expert_file_name, 'r') as expert_file:
            experiment_name = expert_file.readline()[:-1]
            n_time_steps = int(expert_file.readline()) # number of time steps of current experiment
            total_time_steps = total_time_steps + n_time_steps

            for time_step in range(n_time_steps):
                line = expert_file.readline()
                fields = line.split()
                experiment_time = int(fields[0])
                current_action = fields[-1]

                if current_action not in task_actions_dict:
                    logging.error("Filename: %s", expert_file_name)
                    logging.error("Line: %d", time_step+3)
                    logging.error("current_action %s not recognized", current_action)
                    sys.exit()

                task_state_vector = State(*map(int, fields[1:-1]))

                if time_step == 0: # check for valid start state
                    if task_state_vector not in possible_task_start_states:
                        logging.error("expert_file_name: %s", expert_file_name)
                        logging.error("Line: %d", time_step+3)
                        logging.error("State: \n%s", task_state_print(task_state_vector))
                        logging.error("Not valid start state!")
                        sys.exit()
                else:
                    pass
            #print "Experiment Name: ", experiment_name
            #print "Time steps: ", n_time_steps

    #print "Number of files read: ", n_files_read
    #print "Total number of time steps: ", total_time_steps


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
    #mu_e = np.zeros(n_state_vars + len(task_actions_dict))

    #for expert_file_name in glob.glob(os.path.join(data_files_path, '*.txt')):
        #n_files = n_files + 1
        #with open(expert_file_name, 'r') as in_file:
            #e_name = in_file.readline()[:-1] # experiment name
            #n_steps = int(in_file.readline()) # number of time steps of current experiment
            #total_steps = total_steps + n_steps

            #for i in range(n_steps):
                #line = in_file.readline()
                #fields = line.split()
                #e_time = int(fields[0]) # time taken in seconds by current experiment
                #current_action = fields[-1]

                #if current_action not in task_actions_dict:
                    #logging.error("expert_file_name: %s", e_name)
                    #logging.error("Line: %d", i+3)
                    #logging.error("current_action %s not recognized", current_action)
                    #sys.exit()

                #task_state_vector = map(int, fields[1:-1])
                #mu_e = mu_e + get_phi(task_state_vector, current_action)
                #task_state = State(*task_state_vector)

                #if i == 0:
                    #if task_state not in task_start_states:
                        #logging.error("expert_file_name: %s", e_name)
                        #logging.error("Line: %d", i+3)
                        #logging.error("State: %s", str(task_state))
                        #logging.error("Not valid start state!")
                        #sys.exit()
                #else:
                    #if task_state not in task_states:
                        #logging.error("expert_file_name: %s", e_name)
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
    s = ['State Explanation:\n']
    s.append('\tNumber of robot\'s boxes: ' + str(task_state.n_r) + '\n')
    s.append('\tNumber of teammate\'s boxes: ' + str(task_state.n_h) + '\n')
    s.append('\tIs robot transferring? : ' + str(bool(task_state.t_r)) + '\n')
    s.append('\tIs teammate transferring? : ' + str(bool(task_state.t_h)) + '\n')
    s.append('\tIs robot holding its box? : ' + str(bool(task_state.b_r)) + '\n')
    s.append('\tIs robot holding teammate\'s box? : ' + str(bool(task_state.b_h)) + '\n')
    s.append('\tHas the task completed? ' + str(bool(task_state.e)) + '\n')
    return ''.join(s)

