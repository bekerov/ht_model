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
task_data_file = "data/task_data.pickle"
#states_file_path = "../data/states.pickle"
#task_data_path = "../data/task_data.pickle"

n_state_vars = 7
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


def generate_task_state_action_space():
    """Function to generate all state and action space for the box color sort task
    """
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

    # generate the task_state_action_map which is a matrix SxA having value one for those action's indices which are
    # valid for that task_state
    task_state_action_map = np.empty((0, n_action_vars))
    for task_state in task_states:
        # get the indices of the valid actions for the task_state from task_actions
        actions = [task_actions[_][0] for _ in get_valid_actions(State(*task_state.tolist()))]

        # create a row for the current task_state
        current_task_state_map = np.zeros(n_action_vars)
        np.put(current_task_state_map, actions, 1)

        # add the row to the matrix
        task_state_action_map = np.vstack((task_state_action_map, current_task_state_map))

    logging.info("Total number states (after pruning) for box color sort task: %d", len(task_states))
    logging.info("start states length: %d", len(possible_task_start_states))
    return task_states, possible_task_start_states, task_state_action_map

#def get_phi(task_state_vector, current_action):
    #""" Function to return the feature vector given the state vector and action.
    #Arg:
        #List: state vector
        #String: current action
    #Return
        #np_array: phi
    #"""
    #state_feature = [1 if task_state_vector[0] else 0] + [1 if task_state_vector[1] else 0] + task_state_vector[2:]
    #action_feature = [1 if action == current_action else 0 for action in list(task_actions)]
    #feature_vector = state_feature + action_feature
    #return np.array(feature_vector)

#def generate_phi(task_states):
    #phi = np.empty((0, (n_state_vars + len(task_actions))))
    #for task_state in task_states:
        #for task_action in task_actions:
            #phi = np.vstack((phi, get_phi(list(task_state), task_action)))
    #return phi

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

#def state_print(state):
    #"""Function to pretty print the state of the task with elaborate explanations
    #Arg:
        #State: state of the task
    #Returns:
        #string: Complete explanation of the state returned as a string which can be printed
    #"""
    #s = ['State Explanation:\n']
    #s.append('\tNumber of robot\'s boxes: ' + str(state.n_r) + '\n')
    #s.append('\tNumber of teammate\'s boxes: ' + str(state.n_h) + '\n')
    #s.append('\tIs robot transferring? : ' + str(bool(state.t_r)) + '\n')
    #s.append('\tIs teammate transferring? : ' + str(bool(state.t_h)) + '\n')
    #s.append('\tIs robot holding its box? : ' + str(bool(state.b_r)) + '\n')
    #s.append('\tIs robot holding teammate\'s box? : ' + str(bool(state.b_h)) + '\n')
    #s.append('\tHas the task completed? ' + str(bool(state.e)) + '\n')
    #return ''.join(s)

