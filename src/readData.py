#!/usr/bin/env python

import sys
import glob
import os
import time
import cPickle as pickle

from taskSetup import *

def read_data(path):
    possible_states, possible_start_states, possible_actions = load_states("states.pickle")
    state_action = dict()
    visited_states = set()
    total_steps = 0 # cumulative number of time steps of all experiments
    total_time = 0 # cumulative time taken in seconds by all experiments

    for filename in glob.glob(os.path.join(path, '*.txt')):
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
    return visited_states, state_action, time_per_step
