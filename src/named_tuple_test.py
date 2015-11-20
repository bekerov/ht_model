#!/usr/bin/env python

from collections import namedtuple

actions = {
           'W': 'Wait for human',
           'T': 'Take from table',
           'G': 'Give to human',
           'K': 'Keep on table',
           'R': 'Receive from human',
           'X': 'Exit task'
           }

MAX_OBJECTS = 8
MAX_OBJECTS_ACC = MAX_OBJECTS/2

State = namedtuple("State", ['n', 'r_t', 'h_t', 'r_o', 'h_o', 'e'])
start = State(n = 4, r_t = False, h_t = False, r_o = False, h_o = False, e = False)
state_action = {start: ['T']}
print "state = [%d, %d, %d, %d, %d, %d]" % start
print state_action[start]
state_action[start] = 'X'
s = set()
s.add(start)
s.add(start)
print s

