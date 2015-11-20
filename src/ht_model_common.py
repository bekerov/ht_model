#!/usr/bin/env python

import sys
import glob
import os
import random
import time
from termcolor import colored
from collections import namedtuple

actions = {
           'W': 'Wait for teammate',
           'T': 'Take object from table',
           'G': 'Give object to teammate',
           'K': 'Keep object on table',
           'R': 'Receive object from teammate',
           'X': 'Exit task'
           }

MAX_OBJECTS = 8
MAX_OBJECTS_ACC = MAX_OBJECTS/2
State = namedtuple("State", ['n', 'r_t', 'h_t', 'r_o', 'h_o', 'e'])

def get_states_action(path):
    state_action = dict()
    states = set()
    for filename in glob.glob(os.path.join(path, '*.txt')):
        inFile = open(filename, 'r')
        e_name = inFile.readline()
        n_steps = int(inFile.readline())

        for i in range(n_steps):
            line = inFile.readline()
            fields = line.split()
            time_step = int(fields[0])
            action = fields[-1]
            state_vector = map(int, fields[1:-1])
            state = State(n = state_vector[0], r_t = state_vector[1], h_t = state_vector[2], r_o = state_vector[3], h_o = state_vector[4], e = state_vector[5])
            states.add(state)
            if state not in state_action:
                state_action[state] = dict()
            if action in state_action[state]:
                state_action[state][action] = state_action[state][action] + 1
            else:
                state_action[state][action] = 1
            #print "state = [%d, %d, %d, %d, %d, %d]" % state
            #print action
        inFile.close()
    return states, state_action

def print_state(state):
    s = ['Explanation:\n']
    s.append('Number of books on my side: ' + str(state.n) + '\n')
    s.append('Am I transferring? : ' + str(bool(state.r_t)) + '\n')
    s.append('Is teammate transferring? : ' + str(bool(state.h_t)) + '\n')
    s.append('Am I holding my object? : ' + str(bool(state.r_o)) + '\n')
    s.append('Am I holding my teammate''s object? : ' + str(bool(state.h_o)) + '\n')
    s.append('Has the task ended? ' + str(bool(state.e)) + '\n')
    return ''.join(s)


def get_common_policy(state_action):
    policy = dict()
    for state, actions in state_action.items():
        policy[state] = max(actions, key=actions.get)
    return policy

if __name__=='__main__':
    if len(sys.argv) != 2:
        print "Usage: " + sys.argv[0] + " path to data files"
        sys.exit()
    states, state_action = get_states_action(sys.argv[1])
    policy = get_common_policy(state_action)
    while True:
        state = random.choice(tuple(states))
        sys.stdout.write("state = [%d, %d, %d, %d, %d, %d]\n" % state)
        sys.stdout.write(print_state(state))
        print colored("My action: %s" % actions[policy[state]], 'green')
        sys.stdout.write("***********************************\n")
        time.sleep(2)
