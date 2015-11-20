#!/usr/bin/env python

import sys
import glob
import os
from collections import namedtuple

#actions = {
           #'W': 'Wait for human',
           #'T': 'Take from table',
           #'G': 'Give to human',
           #'K': 'Keep on table',
           #'R': 'Receive from human',
           #'X': 'Exit task'
           #}

MAX_OBJECTS = 8
MAX_OBJECTS_ACC = MAX_OBJECTS/2
State = namedtuple("State", ['n', 'r_t', 'h_t', 'r_o', 'h_o', 'e'])

#start = State(n = MAX_OBJECTS/2, r_t = False, h_t = False, r_o = False, h_o = False, e = False)
#state_action = {start: ['T']}
#print "state = [%d, %d, %d, %d, %d, %d]" % start
#print state_action[start]
#state_action[start] = 'X'
#s = set()
#s.add(start)

def get_state_action(filename):
    inFile = open(filename, 'r')
    e_name = inFile.readline()
    n_steps = int(inFile.readline())
    state_action = dict()

    for i in range(n_steps):
        line = inFile.readline()
        fields = line.split()
        time_step = int(fields[0])
        action = fields[-1]
        state_vector = map(int, fields[1:-1])
        state = State(n = state_vector[0], r_t = state_vector[1], h_t = state_vector[2], r_o = state_vector[3], h_o = state_vector[4], e = state_vector[5])
        state_action[state] = dict()
        if action in state_action[state]:
            state_action[state][action] = state_action[state][action] + 1
        else:
            state_action[state][action] = 1
        #print "state = [%d, %d, %d, %d, %d, %d]" % state
    inFile.close()
    return state_action


if __name__=='__main__':
    if len(sys.argv) != 2:
        print "Usage: " + sys.argv[0] + " path to data files"
        sys.exit()
    path = sys.argv[1]
    state_action = dict()
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
            if state not in state_action:
                state_action[state] = dict()
            state_action[state] = dict()
            if action in state_action[state]:
                state_action[state][action] = state_action[state][action] + 1
            else:
                state_action[state][action] = 1
            #print "state = [%d, %d, %d, %d, %d, %d]" % state
        inFile.close()
    print state_action
    #filename = '../data/sample/pair_1_2.txt'
    #print get_state_action(filename)

