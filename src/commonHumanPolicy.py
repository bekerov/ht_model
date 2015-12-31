#!/usr/bin/env python

import sys
import os
import glob
import time
import random
import cPickle as pickle

from termcolor import colored

import taskSetup as ts
import helperFunctions as hf

"""This module creates a common policy for the box color sort task
   based on the most frequent actions taken by humans given a particular
   state
"""

def read_data():
    """Function to read the data files that contains the trajectories of human-human teaming for box color sort task.
    Arg:
        arg1: Path to directory containing the files
    Returns:
        set: all states visited
        dict: dict of dicts mapping states to actions to frequency of that action
        float: time per time step in seconds
    """
    possible_states, possible_start_states, possible_actions = ts.load_states()
    state_action = dict()
    visited_states = set()
    total_steps = 0 # cumulative number of time steps of all experiments
    total_time = 0 # cumulative time taken in seconds by all experiments
    n_files = 0

    for filename in glob.glob(os.path.join(ts.data_files_path, '*.txt')):
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

                if action not in ts.permitted_actions:
                    print "Filename: ", e_name
                    print "Line: ", i+3
                    print "Action %s not recognized" % action
                    sys.exit()

                state_vector = map(int, fields[1:-1])
                state = ts.State(*state_vector)

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

def get_common_policy(state_action):
    """Function to extract the common policy given the state action mapping from human-human task
    Arg:
        dict: dict of dict mapping states to actions which in turn is mapped to its frequency of use
    Returns:
        dict: mapping state to the most frequent action of that particular state
    """
    policy = dict()
    for possible_state, possible_actions in taken_actions.items():
        actions = {k: float(v)/total for total in (sum(possible_actions.values()),) for k, v in possible_actions.items()}
        policy[possible_state] = actions
    return policy

if __name__=='__main__':
    visited_states, taken_actions, time_per_step = read_data()
    policy = get_common_policy(taken_actions)
    with open(ts.policy_file_path, "wb") as p_file:
            pickle.dump(policy, p_file)
    print "Total number of visited states: ", len(visited_states)
    print "Seconds per time step: ", round(time_per_step, 2)
    while True:
        state = random.choice(tuple(visited_states))
        print state
        print ts.state_print(state)
        print colored("Robot\'s action: %s" % ts.permitted_actions[hf.softmax_select_action(policy[state])], 'green')
        print "**********************************************************"
        user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        if user_input.upper() == 'Q':
            break;
        print "**********************************************************"
