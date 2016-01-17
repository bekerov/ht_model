#!/usr/bin/env python

import sys
import logging
import random
import pprint
import numpy as np

from termcolor import colored

import qLearning as ql
import featureExpectation as mu
import simulationFunctions as sf

from loadTaskParams import *
from helperFuncs import *

# set up logging
#logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
logging.basicConfig(format='')
lgr = logging.getLogger("alActionDistribution.py")
lgr.setLevel(level=logging.DEBUG)

def compute_mu_bar_curr(mu_e, mu_bar_prev, mu_curr):
    """Function to compute mu_bar_current using the projection forumula from Abbeel and Ng's paper page 4
    """
    x = mu_curr - mu_bar_prev
    y = mu_e - mu_bar_prev
    mu_bar_curr = mu_bar_prev + (np.dot(x.T, y)/np.dot(x.T, x)) * x

    return mu_bar_curr

def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    mu_e_normalized = expert_feature_expectation/np.linalg.norm(expert_feature_expectation, ord = 1)

    lgr.info("First iteration does not call qlearning")
    i = 1
    while True:
        lgr.debug("%s", colored("*********************************** Iteration %d *********************************************" % (i), 'white', attrs = ['bold']))
        if i == 1:
            r1_state_action_dist = compute_random_state_action_distribution()
            r2_state_action_dist = compute_random_state_action_distribution()
        else:
            r1_state_action_dist, r2_state_action_dist = ql.team_q_learning(r1_state_action_dist, r1_reward, r2_state_action_dist, r2_reward, n_episodes = 10, alpha = 0.5, t = 0.5)

        mu_curr_r1, mu_curr_r2 = mu.compute_normalized_feature_expectation(r1_state_action_dist, r2_state_action_dist)

        if i == 1:
            mu_bar_curr_r1 = mu_curr_r1
            mu_bar_curr_r2 = mu_curr_r2
        else:
            mu_bar_curr_r1 = compute_mu_bar_curr(mu_e_normalized, mu_bar_prev_r1, mu_curr_r1)
            mu_bar_curr_r2 = compute_mu_bar_curr(mu_e_normalized, mu_bar_prev_r2, mu_curr_r2)

        # update the weights
        r1_w = mu_e_normalized - mu_curr_r1
        r2_w = mu_e_normalized - mu_curr_r2

        r1_t = np.linalg.norm(r1_w)
        r2_t = np.linalg.norm(r2_w)

        r1_reward = np.reshape(np.dot(feature_matrix, r1_w), (n_states, ts.n_action_vars))
        r2_reward = np.reshape(np.dot(feature_matrix, r2_w), (n_states, ts.n_action_vars))

        if i % 10 == 1:
            lgr.debug("%s\n", colored("mu_e_normalized = %s" % (mu_e_normalized), 'green', attrs = ['bold']))

            lgr.debug("%s", colored("mu_curr_r1 = %s" % (mu_curr_r1), 'red', attrs = ['bold']))
            lgr.debug("%s\n", colored("mu_curr_r2 = %s" % (mu_curr_r2), 'cyan', attrs = ['bold']))

            lgr.debug("%s", colored("mu_bar_curr_r1 = %s" % (mu_bar_curr_r1), 'red', attrs = ['bold']))
            lgr.debug("%s\n", colored("mu_bar_curr_r2 = %s" % (mu_bar_curr_r2), 'cyan', attrs = ['bold']))

            lgr.debug("%s", colored("r1_w 1-norm = %s" % (np.linalg.norm(r1_w, ord = 1)), 'red', attrs = ['bold']))
            lgr.debug("%s\n", colored("r2_w 1-norm = %s" % (np.linalg.norm(r2_w, ord = 1)), 'cyan', attrs = ['bold']))

            lgr.debug("%s", colored("r1_t = %s" % (r1_t), 'red', attrs = ['bold']))
            lgr.debug("%s", colored("r2_t = %s" % (r2_t), 'cyan', attrs = ['bold']))

            lgr.debug("%s", colored("*********************************** End of Iteration %d **************************************" % (i), 'white', attrs = ['bold']))
            user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
            if user_input.upper() == 'Q':
                sys.exit()

        i = i + 1
        mu_bar_prev_r1 = mu_bar_curr_r1
        mu_bar_prev_r2 = mu_bar_curr_r2

if __name__=='__main__':
    main()

