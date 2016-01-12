#!/usr/bin/env python

import logging
import random
import numpy as np

from termcolor import colored

import taskSetup as ts
import simulationFunctions as sf

task_states, task_start_states, task_state_action_map, phi_matrix = ts.load_state_data()
expert_visited_states, expert_state_action_map, mu_e, n_files = ts.read_task_data()

def main():
    print expert_visited_states

if __name__=='__main__':
    main()
