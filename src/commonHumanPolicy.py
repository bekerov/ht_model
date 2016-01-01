#!/usr/bin/env python

import sys
import os
import glob
import time
import random
import logging
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
                    logging.error("Filename: %s", e_name)
                    logging.error("Line: %d", i+3)
                    logging.error("Action %s not recognized", action)
                    sys.exit()

                task_state_vector = map(int, fields[1:-1])
                task_state = ts.State(*task_state_vector)

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
    logging.info("Total files read: %d", n_files)
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
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s-%(levelname)s: %(message)s')
    task_states, task_start_states, task_state_action_map = ts.load_states()
    expert_visited_states, expert_state_action_map, time_per_step = read_data()
    expert_policy = get_common_policy()
    if not os.path.isfile(ts.policy_file_path):
        logging.info("Generating %s file", ts.policy_file_path)
        with open(ts.policy_file_path, "wb") as p_file:
                pickle.dump(expert_policy, p_file)
    logging.info("Total number of visited states: %d", len(expert_visited_states))
    logging.info("Seconds per time step: %f", round(time_per_step, 2))
    start_state = random.choice(tuple(task_start_states))
    pi_1 = expert_policy
    pi_2 = expert_policy
    nactions = sf.run_simulation(pi_1, pi_2, start_state)
    #state_r1 = random.choice(tuple(task_start_states))
    #state_r2 = state_r1
    #pi_1 = expert_policy
    #pi_2 = expert_policy
    #nactions = 0
    #while True:
        #nactions = nactions + 1
        #action_r1 = sf.softmax_select_action(pi_1[state_r1])
        #action_r2 = sf.softmax_select_action(pi_2[state_r2])
        #logging.debug("%s", colored("state_r1 before: %s" % str(state_r1), 'red'))
        #logging.debug("%s", colored("action_r1: %s" % ts.task_actions[action_r1], 'red'))
        #logging.debug("%s", colored("state_r2 before: %s" % str(state_r2), 'cyan'))
        #logging.debug("%s", colored("action_r2: %s" % ts.task_actions[action_r2], 'cyan'))
        #if action_r1 == 'X' or action_r2 == 'X':
            #break
        #state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
        #state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
        #logging.debug("%s", colored("state_r1 after: %s" % str(state_r1), 'red'))
        #logging.debug("%s", colored("state_r2 after: %s" % str(state_r2), 'cyan'))
        #logging.debug("******************************************************************************")
        #logging.debug("******************************************************************************")

    print "Total number of actions by agents using expert policy is %d" % nactions
