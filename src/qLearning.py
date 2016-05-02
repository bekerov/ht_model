#!/usr/bin/env python

import sys
import logging
import random
import pprint
import numpy as np

from termcolor import colored

import simulationFunctions as sf

from loadTaskParams import *
from helperFuncs import *

#logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
logging.basicConfig(format='')
lgr = logging.getLogger("qLearning.py")
lgr.setLevel(level=logging.INFO) # change level to debug for output

def team_q_learning(r1_state_action_dist, r1_reward, r2_state_action_dist, r2_reward, n_episodes = 10, temp = 1.0):
    """Function that runs the q learning algorithm for both the agents and returns the action_distribution (after softmaxing it)
    """
    lgr.debug("%s", colored("                                        TEAM_Q_LEARNING                     ", 'white', attrs = ['bold']))

    r1_q = np.zeros((n_states, ts.n_action_vars))
    r2_q = np.zeros((n_states, ts.n_action_vars))

    # for actions that cannot be taken in particular states, set q value to be -inf so that, that action will never be chosen
    r1_q[state_action_space == -np.inf] = -np.inf
    r2_q[state_action_space == -np.inf] = -np.inf

    gamma = 0.99
    alpha = 1.0
    alpha_dec_factor = 0.99
    alpha_lb = 0.1

    for episode in range(n_episodes):
        lgr.debug("%s", colored("************************************* Episode %d ****************************************************" % (episode+1), 'white', attrs = ['bold']))
        start_state = random.choice(task_start_states_list)
        r1_state_idx = task_states_list.index(start_state)
        r2_state_idx = r1_state_idx
        r1_state_tup = start_state
        r2_state_tup = start_state
        alpha = alpha_lb if alpha <= alpha_lb else alpha * alpha_dec_factor
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

            # get max action index for both agents
            r1_max_action_idx = r1_q[r1_state_prime_idx].argmax()
            r2_max_action_idx = r2_q[r2_state_prime_idx].argmax()

            lgr.debug("%s", colored("r1_max_action = %s, action_idx = %d" % (ts.task_actions_index[r1_max_action_idx], r1_max_action_idx), 'red', attrs = ['bold']))
            lgr.debug("%s\n", colored("r2_max_action = %s, action_idx = %d" % (ts.task_actions_index[r2_max_action_idx], r2_max_action_idx), 'cyan', attrs = ['bold']))

            lgr.debug("%s\n", colored("################################## Q Value Before Update #####################################", 'white', attrs = ['bold']))
            lgr.debug("%s", colored("r1_q[%d][%d] = %f" % (r1_state_idx, r1_action_idx, r1_q[r1_state_idx][r1_action_idx]), 'red', attrs = ['bold']))
            lgr.debug("%s\n", colored("r2_q[%d][%d] = %f" % (r2_state_idx, r2_action_idx, r2_q[r2_state_idx][r2_action_idx]), 'cyan', attrs = ['bold']))

            r1_q[r1_state_idx][r1_action_idx] = r1_q[r1_state_idx][r1_action_idx] + alpha * (r1_reward[r1_state_idx][r1_action_idx] +
                    gamma * r1_q[r1_state_prime_idx][r1_max_action_idx] - r1_q[r1_state_idx][r1_action_idx])
            r2_q[r2_state_idx][r2_action_idx] = r2_q[r2_state_idx][r2_action_idx] + alpha * (r2_reward[r2_state_idx][r2_action_idx] +
                    gamma * r2_q[r2_state_prime_idx][r2_max_action_idx] - r2_q[r2_state_idx][r2_action_idx])

            lgr.debug("%s\n", colored("################################## Q Value After Update #####################################", 'white', attrs = ['bold']))
            lgr.debug("%s", colored("r1_q[%d][%d] = %f" % (r1_state_idx, r1_action_idx, r1_q[r1_state_idx][r1_action_idx]), 'red', attrs = ['bold']))
            lgr.debug("%s\n", colored("r2_q[%d][%d] = %f" % (r2_state_idx, r2_action_idx, r2_q[r2_state_idx][r2_action_idx]), 'cyan', attrs = ['bold']))

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

        logging.debug("%s", colored("************************************* End of Episode %d ****************************************************" % (episode+1), 'white', attrs = ['bold']))

    return softmax(r1_q, temp), softmax(r2_q, temp)
