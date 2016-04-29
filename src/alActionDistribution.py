#!/usr/bin/env python

import sys
import logging
import random
import pprint
import numpy as np
import cPickle as pickle

from termcolor import colored
from scipy import stats

import qLearning as ql
import featureExpectation as mu
import simulationFunctions as sf

from loadTaskParams import *
from helperFuncs import *

# set up logging
logging.basicConfig(format='')
lgr = logging.getLogger("alActionDistribution.py")
lgr.setLevel(level=logging.INFO)

MAX_BEST_STATE_ACTION_DISTS = 10

def locate_min(a):
    """Function to compute the smallest in an array and get all the indicies with the
    smallest value
    """
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) if smallest == element]

def compute_mu_bar_curr(mu_e, mu_bar_prev, mu_curr):
    """Function to compute mu_bar_current using the projection forumula from Abbeel and Ng's paper page 4
    """
    x = mu_curr - mu_bar_prev
    y = mu_e - mu_bar_prev
    mu_bar_curr = mu_bar_prev + (np.dot(x.T, y)/np.dot(x.T, x)) * x

    return mu_bar_curr

def simulate_learned_state_action_distribution():
    """Function to simulate AL learned action distribution for both the agents
    """
    lgr.info("Loading best distributions for all start states from best_dists.pickle")
    with open("best_dists.pickle", "r") as best_dists_file:
        r1_best_dists = pickle.load(best_dists_file)
        r2_best_dists = pickle.load(best_dists_file)

    start_state = random.choice(task_start_states_list)
    r1_learned_state_action_distribution_dict = convert_to_dict_from_numpy(random.choice(r1_best_dists[start_state]))
    r2_learned_state_action_distribution_dict = convert_to_dict_from_numpy(random.choice(r2_best_dists[start_state]))
    lgr.info("%s", colored("Total number of actions by agents using least actions policy is %d" % sf.run_simulation(random.choice(r1_best_dists[start_state]), random.choice(r2_best_dists[start_state]), start_state), 'white', attrs = ['bold']))
    lgr.info("%s", colored("Start State: %s" % str(start_state), 'white', attrs = ['bold']))

def learn_agent_dists():
    mu_e_normalized = expert_feature_expectation/np.linalg.norm(expert_feature_expectation, ord = 1)
    epsilon = 0.075
    temp = 0.9
    temp_dec_factor = 0.95
    temp_lb = 0.2
    r1_t = np.inf
    r2_t = np.inf
    r1_initial_state_action_dist = compute_uniform_state_action_distribution()
    r2_initial_state_action_dist = compute_uniform_state_action_distribution()

    r1_dists = list()
    r2_dists = list()

    lgr.debug("%s", colored("First iteration does not call qlearning", 'white', attrs = ['bold']))
    i = 1
    while max(r1_t, r2_t) > epsilon:
        if i == 1:
            r1_state_action_dist = r1_initial_state_action_dist
            r2_state_action_dist = r2_initial_state_action_dist
        else:
            # get a random number of episodes for each run
            n_episodes = random.randint(n_experiments, 3*n_experiments)
            r1_state_action_dist, r2_state_action_dist = ql.team_q_learning(r1_state_action_dist, r1_reward, r2_state_action_dist, r2_reward, n_episodes = n_episodes, temp = temp)
            temp = temp_lb if temp <= temp_lb else temp * temp_dec_factor

        r1_dists.append(r1_state_action_dist)
        r2_dists.append(r1_state_action_dist)

        if i % 50 == 1:
            lgr.info("%s", colored("*********************************** Iteration %d *********************************************" % (i), 'white', attrs = ['bold']))
            lgr.info("%s", colored("r1_t = %s" % (r1_t), 'red', attrs = ['bold']))
            lgr.info("%s", colored("r2_t = %s" % (r2_t), 'cyan', attrs = ['bold']))
            lgr.info("%s", colored("max(r1_t, r2_t) = %s" % (max(r1_t, r2_t)), 'green', attrs = ['bold']))

        mu_curr_r1, mu_curr_r2 = mu.compute_normalized_feature_expectation(r1_state_action_dist, r2_state_action_dist)

        if i == 1:
            mu_bar_curr_r1 = mu_curr_r1
            mu_bar_curr_r2 = mu_curr_r2
        else:
            mu_bar_curr_r1 = compute_mu_bar_curr(mu_e_normalized, mu_bar_prev_r1, mu_curr_r1)
            mu_bar_curr_r2 = compute_mu_bar_curr(mu_e_normalized, mu_bar_prev_r2, mu_curr_r2)

        # update the weights
        if r1_t > epsilon:
            r1_w = mu_e_normalized - mu_bar_curr_r1
        if r2_t > epsilon:
            r2_w = mu_e_normalized - mu_bar_curr_r2

        r1_t = np.linalg.norm(r1_w)
        r2_t = np.linalg.norm(r2_w)

        r1_reward = np.reshape(np.dot(feature_matrix, r1_w), (n_states, ts.n_action_vars))
        r2_reward = np.reshape(np.dot(feature_matrix, r2_w), (n_states, ts.n_action_vars))

        if i % 500 == 1:
            if lgr.getEffectiveLevel() == logging.DEBUG:
                lgr.debug("%s\n", colored("mu_e_normalized = %s" % (mu_e_normalized), 'green', attrs = ['bold']))
                lgr.debug("%s", colored("mu_curr_r1 = %s" % (mu_curr_r1), 'red', attrs = ['bold']))
                lgr.debug("%s\n", colored("mu_curr_r2 = %s" % (mu_curr_r2), 'cyan', attrs = ['bold']))

                lgr.debug("%s", colored("mu_bar_curr_r1 = %s" % (mu_bar_curr_r1), 'red', attrs = ['bold']))
                lgr.debug("%s\n", colored("mu_bar_curr_r2 = %s" % (mu_bar_curr_r2), 'cyan', attrs = ['bold']))
                lgr.debug("%s", colored("r1_w 1-norm = %s" % (np.linalg.norm(r1_w, ord = 1)), 'red', attrs = ['bold']))
                lgr.debug("%s\n", colored("r2_w 1-norm = %s" % (np.linalg.norm(r2_w, ord = 1)), 'cyan', attrs = ['bold']))

                lgr.debug("%s", colored("r1_t = %s" % (r1_t), 'red', attrs = ['bold']))
                lgr.debug("%s", colored("r2_t = %s" % (r2_t), 'cyan', attrs = ['bold']))
                lgr.debug("%s", colored("max(r1_t, r2_t) = %s" % (max(r1_t, r2_t)), 'green', attrs = ['bold']))

                lgr.debug("%s", colored("*********************************** End of Iteration %d **************************************" % (i), 'white', attrs = ['bold']))
                user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
                if user_input.upper() == 'Q':
                    break
        i = i + 1
        mu_bar_prev_r1 = mu_bar_curr_r1
        mu_bar_prev_r2 = mu_bar_curr_r2

    lgr.info("%s", colored("*********************************** Iteration %d *********************************************" % (i-1), 'white', attrs = ['bold']))
    lgr.info("%s", colored("r1_t = %s" % (r1_t), 'red', attrs = ['bold']))
    lgr.info("%s", colored("r2_t = %s" % (r2_t), 'cyan', attrs = ['bold']))
    lgr.info("%s", colored("max(r1_t, r2_t) = %s" % (max(r1_t, r2_t)), 'green', attrs = ['bold']))
    lgr.info("%s", colored("Number of iterations: %d" % (i-1), 'white', attrs = ['bold']))

    with open("dists.pickle", "wb") as dists_file:
        pickle.dump(r1_dists, dists_file)
        pickle.dump(r2_dists, dists_file)

    return r1_dists, r2_dists

if __name__=='__main__':
    if len(sys.argv) < 2:
        lgr.error("Usage: %s s(simulate)|l(learn)", sys.argv[0])
        sys.exit()
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)

    if sys.argv[1] == 'l':
        learn_agent_dists()
    else:
        simulate_learned_state_action_distribution()

