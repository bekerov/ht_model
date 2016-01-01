#!/usr/bin/env python

import sys
import os
import glob
import time
import random
import cPickle as pickle

from termcolor import colored

import taskSetup as ts
import simulationFunctions as sf

"""This module creates a common policy for the box color sort task
   based on the most frequent actions taken by humans given a particular
   state
"""

def read_data():
    """Function to read the data files that contains the trajectories of human-human teaming for box color sort task.
    Returns:
        set: all states visited
        dict: dict of dicts mapping states to actions to frequency of that action
        float: time per time step in seconds
    """
    expert_state_action_map = dict()
    expert_visited_states = set()
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

                if action not in ts.task_actions:
                    print "Filename: ", e_name
                    print "Line: ", i+3
                    print "Action %s not recognized" % action
                    sys.exit()

                task_state_vector = map(int, fields[1:-1])
                task_state = ts.State(*task_state_vector)

                if i == 0:
                    if task_state not in task_start_states:
                        print "Filename: ", e_name
                        print "Line: ", i+3
                        print "State: ", state
                        print "Not valid start state!"
                        sys.exit()
                else:
                    if task_state not in task_states:
                        print "Filename: ", e_name
                        print "Line: ", i+3
                        print "State: ", state
                        print "Not valid state!"
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
    print "Total files read: ", n_files
    return expert_visited_states, expert_state_action_map, time_per_step

def get_common_policy():
    """Function to extract the common policy given the state action mapping from human-human task
    Returns:
        dict: mapping state to the most frequent action of that particular state
    """
    policy = dict()
    for task_state, state_actions in task_state_action_map.items():
        actions = dict()
        for action in state_actions:
            actions[action] = 1.0/len(state_actions)
        policy[task_state] =actions

    for task_state, expert_actions in expert_state_action_map.items():
        actions = {k: float(v)/total for total in (sum(expert_actions.values()),) for k, v in expert_actions.items()}
        policy[task_state] = actions
    return policy

if __name__=='__main__':
    task_states, task_start_states, task_state_action_map = ts.load_states()
    expert_visited_states, expert_state_action_map, time_per_step = read_data()
    expert_policy = get_common_policy()
    #for task_state, expert_action in expert_policy.items():
        #print task_state, expert_action

    with open(ts.policy_file_path, "wb") as p_file:
            pickle.dump(expert_policy, p_file)
    #print "Total number of visited states: ", len(expert_visited_states)
    #print "Seconds per time step: ", round(time_per_step, 2)
    #while True:
        #state = random.choice(tuple(expert_visited_states))
        #print state
        #print ts.state_print(state)
        #print colored("Robot\'s action: %s" % ts.task_actions[sf.softmax_select_action(expert_policy[state])], 'green')
        #print "**********************************************************"
        #user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        #if user_input.upper() == 'Q':
            #break;
        #print "**********************************************************"
