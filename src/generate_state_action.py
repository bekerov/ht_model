#!/usr/bin/env python

import sys
import glob
import os
import time

from collections import namedtuple
from pprint import pprint
from itertools import product
from random import choice
from termcolor import colored

permitted_actions = {
        'WG': 'Wait for teammate to receive',
        'WS': 'Wait for state change',
        'TR': 'Take robot\'s box from table',
        'TH': 'Take teammate\'s box from table',
        'G': 'Give box to teammate',
        'K': 'Keep box on table',
        'R': 'Receive box from teammate',
        'X': 'Exit task'
        }

MAX_BOXES = 8
MAX_BOXES_ACC = MAX_BOXES/2

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

def generate_states():
    def is_valid_state(state):
        if state.e == 1:
            # task is done, so other elements of the state vector cannot be non-zero
            return all(v == 0 for v in state[0:-1])
        if (state.n_r + state.n_h) > MAX_BOXES_ACC:
            # total number of boxes on robot's side must be less than or equal to max number of boxes
            # accessable or zero when robot's side is sorted and robot is waiting for teammate
            return False
        if state.t_r == 1 and state.b_h != 1:
            # if the robot is transferring a box, then it must be holding the
            # teammate's box for the state to be valid
            return False
        if (state.n_r + state.n_h) == MAX_BOXES_ACC and state.b_h == 1:
            # if robot has all its accessible boxes then,
            # if the robot gets a box, the box was received from a teammate transfer
            # thus box in hand cannot be teammate's box
            return False
        return True
    states = set()
    vals = [
            [v for v in range(MAX_BOXES_ACC+1)],
            [v for v in range(MAX_BOXES_ACC+1)],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1]
            ]
    for s in product(*vals):
        state = State(*s)
        if is_valid_state(state):
            states.add(state)

    start_states = set()
    for state in states:
        if (state.n_r + state.n_h) == MAX_BOXES_ACC and all (v == 0 for v in state[2:]):
            start_states.add(state)

    return frozenset(states), frozenset(start_states)

def generate_actions(states):
    def generate_permitted_actions(state):
        actions = set()
        if all(v == 0 for v in state):
            # robot is done with its part and can only wait for teammate to change
            # state
            actions.add('WS')
        if state.n_r:
            # if there are robot's boxes on the robots side, take it
            actions.add('TR')
        if state.n_h:
            # if there are human's boxes on the robots side, take it
            actions.add('TH')
        # if (state.n_r + state.n_h):
            # actions.add('T')
        if state.b_h == 1 and state.t_r == 1:
            # if the robot is transferring it can wait for the teammate to receive
            actions.add('WG')
        if state.t_h == 1:
            # if the teammate is transferring then the robot can receive
            actions.add('R')
        if state.b_r == 1:
            # if the robot is holding its box, it can keep
            actions.add('K')
        if state.b_h == 1 and state.t_r == 0:
            # if the robot is holding teammate's box, it can give
            actions.add('G')
        if state.e == 1:
            # if task is done, robot can exit
            actions.add('X')
        return actions

    state_action = dict([ (elem, None) for elem in states ])
    for state in state_action:
        state_action[state] = generate_permitted_actions(state)

    return state_action

possible_states, possible_start_states = generate_states()
possible_actions = generate_actions(possible_states)

def state_print(state):
    s = ['State Explanation:\n']
    s.append('\tNumber of robot\'s boxes: ' + str(state.n_r) + '\n')
    s.append('\tNumber of teammate\'s boxes: ' + str(state.n_h) + '\n')
    s.append('\tIs robot transferring? : ' + str(bool(state.t_r)) + '\n')
    s.append('\tIs teammate transferring? : ' + str(bool(state.t_h)) + '\n')
    s.append('\tIs robot holding its box? : ' + str(bool(state.b_r)) + '\n')
    s.append('\tIs robot holding teammate\'s box? : ' + str(bool(state.b_h)) + '\n')
    s.append('\tHas the task completed? ' + str(bool(state.e)) + '\n')
    return ''.join(s)

def simulate_next_state(state, r_action, h_action):
    return state

def read_data_files(path):
    state_action = dict()
    visited_states = set()
    total_steps = 0 # cumulative number of time steps of all experiments
    total_time = 0 # cumulative time taken in seconds by all experiments

    for filename in glob.glob(os.path.join(path, '*.txt')):
        inFile = open(filename, 'r')
        e_name = inFile.readline()[:-1] # experiment name
        n_steps = int(inFile.readline()) # number of time steps of current experiment
        total_steps = total_steps + n_steps

        for i in range(n_steps):
            line = inFile.readline()
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
        inFile.close()
        total_time = total_time + e_time/2.0 # dividing by 2.0 since, all the videos were stretched twice for manual processing

    time_per_step = total_time / total_steps
    return visited_states, state_action, time_per_step

def get_common_policy(state_action):
    policy = dict()
    for state, actions in state_action.items():
        policy[state] = max(actions, key=actions.get)
    return policy

if __name__=='__main__':
    if len(sys.argv) != 2:
        print "Usage: " + sys.argv[0] + " path to data files"
        sys.exit()
    read_data_files(sys.argv[1])
    visited_states, taken_actions, time_per_step = read_data_files(sys.argv[1])
    policy = get_common_policy(taken_actions)
    while True:
        state = choice(tuple(visited_states))
        print state
        print state_print(state)
        print colored("Robot\'s action: %s" % permitted_actions[policy[state]], 'green')
        sys.stdout.write("**********************************************************\n")
        sys.stdout.write("**********************************************************\n")
        if policy[state] == 'X':
            break
        time.sleep(time_per_step)
    #states, start_states = generate_states()
    #state_action = generate_state_action(states)
    #state = choice(tuple(states))
    #actions = state_action[state]
    #print state
    #print state_print(state)
    #print "***********************************************"
    #print "Allowed Actions:"
    #for i, action in enumerate(actions):
        #print "\t", i+1, permitted_actions[action]

