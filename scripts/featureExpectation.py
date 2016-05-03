#!/usr/bin/env python

import sys
import logging
import random
import numpy as np

from termcolor import colored

import simulationFunctions as sf

from loadTaskParams import *
from helperFuncs import *

#logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
logging.basicConfig(format='')
lgr = logging.getLogger("featureExpectation.py")
lgr.setLevel(level=logging.INFO) # change level to debug for output

def get_feature_idx(state_idx, action_idx):
    """Function to unravel feature_matrix index given state and action index
    """
    return (state_idx * ts.n_action_vars + action_idx)

def compute_normalized_feature_expectation(r1_state_action_dist, r2_state_action_dist):
    """Function to compute the feature expectation of the agents by running the simulation for n_experiments. The feature expectations are normalized (using 1-norm) to bind them within 1
    """
    lgr.debug("%s", colored("                                        COMPUTE_NORMALIZED_FEATURE_EXPECTATION                     ", 'white', attrs = ['bold']))

    r1_feature_expectation = np.zeros(ts.n_state_vars + ts.n_action_vars)
    r2_feature_expectation = np.zeros(ts.n_state_vars + ts.n_action_vars)

    for i in range(n_experiments):
        lgr.debug("%s", colored("************************************* Trial %d ****************************************************" % (i), 'white', attrs = ['bold']))
        start_state = random.choice(task_start_states_list)
        r1_state_idx = task_states_list.index(start_state)
        r2_state_idx = r1_state_idx
        r1_state_tup = start_state
        r2_state_tup = start_state
        step = 1

        while True:
            lgr.debug("%s", colored("************************************* Step %d ****************************************************" % (step), 'white', attrs = ['bold']))
            lgr.debug("%s", colored("r1_state_action_dist = %s" % (r1_state_action_dist[r1_state_idx]), 'red', attrs = ['bold']))
            lgr.debug("%s", colored("r1_state_action_dist = %s" % (r1_state_action_dist[r1_state_idx]), 'cyan', attrs = ['bold']))

            r1_action = select_random_action(r1_state_action_dist[r1_state_idx])
            r2_action = select_random_action(r2_state_action_dist[r2_state_idx])
            r1_action_idx = ts.task_actions_expl[r1_action][0]
            r2_action_idx = ts.task_actions_expl[r2_action][0]

            lgr.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ BEFORE ACTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            lgr.debug("%s", colored("r1_state_tup = %s, state_idx = %d" % (r1_state_tup, r1_state_idx), 'red', attrs = ['bold']))
            lgr.debug("%s", colored("r2_state_tup = %s, state_idx = %d" % (r2_state_tup, r2_state_idx), 'cyan', attrs = ['bold']))
            lgr.debug("%s", colored("r1_action = %s, action_idx = %d" % (r1_action, r1_action_idx), 'red', attrs = ['bold']))
            lgr.debug("%s\n", colored("r2_action = %s, action_idx = %d" % (r2_action, r2_action_idx), 'cyan', attrs = ['bold']))

            if r1_action == 'X' and r2_action == 'X':
                lgr.debug("%s", colored("************************************* End of Step %d ****************************************************" % (step), 'white', attrs = ['bold']))
                break

            r1_state_prime_tup, r2_state_prime_tup = sf.simulate_next_state(r1_action, r1_state_tup, r2_state_tup) # first agent acting

            lgr.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ After 1st Agent Action @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            lgr.debug("%s", colored("r1_state_prime_tup = %s" % str(r1_state_prime_tup), 'red', attrs = ['bold']))
            lgr.debug("%s\n", colored("r2_state_prime_tup = %s" % str(r2_state_prime_tup), 'cyan', attrs = ['bold']))

            r2_state_prime_tup, r1_state_prime_tup = sf.simulate_next_state(r2_action, r2_state_prime_tup, r1_state_prime_tup) # second agent acting
            r1_state_prime_idx = task_states_list.index(r1_state_prime_tup)
            r2_state_prime_idx = task_states_list.index(r2_state_prime_tup)

            lgr.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ After 2nd Agent Action @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            lgr.debug("%s", colored("r1_state_prime_tup = %s, state_idx = %d" % (r1_state_prime_tup, r1_state_prime_idx), 'red', attrs = ['bold']))
            lgr.debug("%s", colored("r2_state_prime_tup = %s, state_idx = %d" % (r2_state_prime_tup, r2_state_prime_idx), 'cyan', attrs = ['bold']))

            r1_feature_expectation = r1_feature_expectation + feature_matrix[get_feature_idx(r1_state_idx, r1_action_idx)]
            r2_feature_expectation = r2_feature_expectation + feature_matrix[get_feature_idx(r2_state_idx, r2_action_idx)]

            # update current states to new states
            r1_state_tup = r1_state_prime_tup
            r2_state_tup = r2_state_prime_tup

            # update the state indices for both agents
            r1_state_idx = task_states_list.index(r1_state_tup)
            r2_state_idx = task_states_list.index(r2_state_tup)

            logging.debug("%s", colored("************************************* End of Step %d ****************************************************" % (step), 'white', attrs = ['bold']))
            step = step + 1
            if lgr.getEffectiveLevel() == logging.DEBUG:
                user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
                if user_input.upper() == 'Q':
                    sys.exit()

        logging.debug("%s", colored("************************************* End of Trial %d ****************************************************" % (i), 'white', attrs = ['bold']))

    r1_feature_expectation = r1_feature_expectation/n_experiments
    r2_feature_expectation = r2_feature_expectation/n_experiments

    return r1_feature_expectation/np.linalg.norm(r1_feature_expectation, ord = 1), r2_feature_expectation/np.linalg.norm(r2_feature_expectation, ord = 1)
