#!/usr/bin/env python

from collections import namedtuple
from pprint import pprint

MAX_BOXES = 8
MAX_BOXES_ACC = MAX_BOXES/2

State = namedtuple("State", ['n_r', 'n_h', 't_r', 't_h', 'b_r', 'b_h', 'e'])

def get_start_states():
    start_states = set()
    for i in range(MAX_BOXES_ACC+1):
        state = State(n_r = i, n_h = MAX_BOXES_ACC - i, t_r = 0, t_h = 0, b_r = 0, b_h = 0, e = 0)
        start_states.add(state)
    return start_states

if __name__=='__main__':
    start_states = get_start_states()
    print start_states
