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
    lgr.info("Loading agent 1 best distribution dictionary")
    with open("r1_agent_best_dists_dict.pickle", "r") as state_action_dict_file:
        r1_best_dists_dict = pickle.load(state_action_dict_file)
    lgr.info("Loading agent 2 best distribution dictionary")
    with open("r2_agent_best_dists_dict.pickle", "r") as state_action_dict_file:
        r2_best_dists_dict = pickle.load(state_action_dict_file)

    start_state = random.choice(task_start_states_list)
    r1_learned_state_action_distribution_dict = random.choice(r1_best_dists_dict[start_state])
    r2_learned_state_action_distribution_dict = random.choice(r2_best_dists_dict[start_state])
    print "Total number of actions by agents using least actions policy is %d" % sf.run_simulation(r1_learned_state_action_distribution_dict, r2_learned_state_action_distribution_dict, random.choice(tuple(task_start_states_list)))

def get_agent_dists():
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

    r1_dists_dict = list()
    r2_dists_dict = list()

    #for state_action_dist in r1_dists:
        #r1_dists_dict.append(extract_state_action_distribution_dict(state_action_dist))
    #for state_action_dist in r2_dists:
        #r2_dists_dict.append(extract_state_action_distribution_dict(state_action_dist))

    #return r1_dists_dict, r2_dists_dict
    return r1_dists, r2_dists

if __name__=='__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    r1_dists, r2_dists = get_agent_dists()
    with open("dists.pickle", "wb") as dists_file:
        pickle.dump(r1_dists, dists_file)
        pickle.dump(r2_dists, dists_file)

    #lgr.info("\n%s", colored("Searching for best policy based on least number of actions for %d trials" % n_trials, 'white', attrs = ['bold']))

    #n_actions_learned = np.zeros(n_trials)
    #best_state_action_dist_indices_dict = dict()
    #for start_state in task_start_states_list:
        #modes = list()
        #for policy_idx in range(len(r1_dists_dict)):
            #r1_learned_state_action_distribution_dict = r1_dists_dict[policy_idx]
            #r2_learned_state_action_distribution_dict = r2_dists_dict[policy_idx]

            #for i in range(n_trials):
                #n_actions_learned[i] = sf.run_simulation(r1_learned_state_action_distribution_dict, r2_learned_state_action_distribution_dict, start_state)
            #modes.append(stats.mode(n_actions_learned)[0][0])
        #smallest, policy_indices = locate_min(modes)
        #best_state_action_dist_indices_dict[start_state] = policy_indices
        #lgr.info("%s", colored("Start State: %s" % str(start_state), 'yellow', attrs = ['bold']))
        #lgr.info("%s", colored("Smallest mode: %d" % smallest, 'white', attrs = ['bold']))
        #lgr.info("%s", colored("Policy indices %s" % str(policy_indices), 'white', attrs = ['bold']))

    #r1_best_dists_dict = dict()
    #r2_best_dists_dict = dict()

    #for start_state, best_state_action_dist_indices_list in best_state_action_dist_indices_dict.items():
        #r1_best_dists_dict[start_state] = list()
        #r2_best_dists_dict[start_state] = list()
        #if len(best_state_action_dist_indices_list) > MAX_BEST_STATE_ACTION_DISTS:
            #best_state_action_dists_indices = random.sample(best_state_action_dist_indices_list, MAX_BEST_STATE_ACTION_DISTS)
        #else:
            #best_state_action_dists_indices = best_state_action_dist_indices_list
        #for idx in best_state_action_dists_indices:
            #r1_best_dists_dict[start_state].append(r1_dists_dict[idx])
            #r2_best_dists_dict[start_state].append(r2_dists_dict[idx])

    #lgr.info("%s", colored("Saving best distribution for both agents as dictionaries in r1_agent_best_dists_dict.pickle and r2_agent_best_dists_dict.pickle" , 'white', attrs = ['bold']))
    #with open("r1_agent_best_dists_dict.pickle", "wb") as agent_best_dists_dict_file:
        #pickle.dump(r1_best_dists_dict, agent_best_dists_dict_file)
    #with open("r2_agent_best_dists_dict.pickle", "wb") as agent_best_dists_dict_file:
        #pickle.dump(r2_best_dists_dict, agent_best_dists_dict_file)
    #simulate_learned_state_action_distribution()

