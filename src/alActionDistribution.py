#!/usr/bin/env python

import logging
import random
import pprint
import numpy as np

from termcolor import colored
import simulationFunctions as sf

from loadTaskParams import *

# set logging level
#logging.basicConfig(level=logging.debug, format='%(asctime)s-%(levelname)s: %(message)s')
logging.basicConfig(level=logging.INFO, format='')

# number of states in task
n_states = len(task_states_list)

# get numpy matrix of task list
state_space = np.zeros((n_states, ts.n_state_vars))
for state_idx, task_state_tup in enumerate(task_states_list):
    state_space[state_idx] = np.array(task_state_tup)

# get numpy value matrix of state action space
# 0 for valid actions and -inf for invalid actions from given state
state_action_space = np.random.rand(n_states, ts.n_action_vars)
state_action_space[:] = -np.inf
for state_idx, task_state_tup in enumerate(task_states_list):
    action_idx = [ts.task_actions_expl[_][0] for _ in task_state_action_dict[task_state_tup]]
    np.put(state_action_space[state_idx], action_idx, 0)

# Q-Learning parameters
alpha = 0.5
gamma = 1.0

def compute_random_state_action_distribution():
    """Function to compute a random state action distribution
    """
    random_state_action_dist = np.random.rand(n_states, ts.n_action_vars)
    random_state_action_dist[state_action_space == -np.inf] = 0
    random_state_action_dist = random_state_action_dist / random_state_action_dist.sum(axis=1).reshape((len(random_state_action_dist), 1)) # normalize each row

    return random_state_action_dist

def extract_best_policy(state_action_dist):
    """Function to extract the policy from q_value matrix using argmax
    """
    policy = dict()
    for state_idx, task_state_tup in enumerate(task_states_list):
        action_idx = np.argmax(state_action_dist[state_idx])
        policy[task_state_tup] = ts.task_actions_index[action_idx]

    return policy

def select_random_action(state_action_vector):
    """Function to select a random action based on the its probability in the state action distribution
    """
    action_idx = np.random.choice(np.arange(state_action_vector.size), p = state_action_vector)
    action = ts.task_actions_index[action_idx]

    return action

def get_feature_idx(state_idx, action_idx):
    return (state_idx * ts.n_action_vars + action_idx)

def compute_normalized_feature_expectation(r1_state_action_dist, r2_state_action_dist):
    """Function to compute the feature expectation of the agents by running the simulation for n_experiments. The feature expectations are normalized (using 1-norm) to bind them within 1
    """
    logging.debug("%s", colored("                                        COMPUTE_NORMALIZED_FEATURE_EXPECTATION                     ", 'white', attrs = ['bold']))

    r1_feature_expectation = np.zeros(ts.n_state_vars + ts.n_action_vars)
    r2_feature_expectation = np.zeros(ts.n_state_vars + ts.n_action_vars)

    for i in range(n_experiments):
        logging.debug("%s", colored("************************************* Trial %d ****************************************************" % (i), 'white', attrs = ['bold']))
        start_state = random.choice(task_start_state_set)
        r1_state_idx = task_states_list.index(start_state)
        r2_state_idx = r1_state_idx
        r1_state_tup = start_state
        r2_state_tup = start_state
        step = 1

        while True:
            logging.debug("%s", colored("************************************* Step %d ****************************************************" % (step), 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_action_dist = %s" % (r1_state_action_dist[r1_state_idx]), 'red', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_action_dist = %s" % (r1_state_action_dist[r1_state_idx]), 'cyan', attrs = ['bold']))

            r1_action = select_random_action(r1_state_action_dist[r1_state_idx])
            r2_action = select_random_action(r2_state_action_dist[r2_state_idx])
            r1_action_idx = ts.task_actions_expl[r1_action][0]
            r2_action_idx = ts.task_actions_expl[r2_action][0]

            logging.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ BEFORE ACTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_tup = %s, state_idx = %d" % (r1_state_tup, r1_state_idx), 'red', attrs = ['bold']))
            logging.debug("%s", colored("r2_state_tup = %s, state_idx = %d" % (r2_state_tup, r2_state_idx), 'cyan', attrs = ['bold']))
            logging.debug("%s", colored("r1_action = %s, action_idx = %d" % (r1_action, r1_action_idx), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_action = %s, action_idx = %d" % (r2_action, r2_action_idx), 'cyan', attrs = ['bold']))

            if r1_action == 'X' and r2_action == 'X':
                logging.debug("%s", colored("************************************* End of Step %d ****************************************************" % (step), 'white', attrs = ['bold']))
                break

            r1_state_prime_tup, r2_state_prime_tup = sf.simulate_next_state(r1_action, r1_state_tup, r2_state_tup) # first agent acting

            logging.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ After 1st Agent Action @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_prime_tup = %s" % str(r1_state_prime_tup), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_state_prime_tup = %s" % str(r2_state_prime_tup), 'cyan', attrs = ['bold']))

            r2_state_prime_tup, r1_state_prime_tup = sf.simulate_next_state(r2_action, r2_state_prime_tup, r1_state_prime_tup) # second agent acting
            r1_state_prime_idx = task_states_list.index(r1_state_prime_tup)
            r2_state_prime_idx = task_states_list.index(r2_state_prime_tup)

            logging.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ After 2nd Agent Action @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_prime_tup = %s, state_idx = %d" % (r1_state_prime_tup, r1_state_prime_idx), 'red', attrs = ['bold']))
            logging.debug("%s", colored("r2_state_prime_tup = %s, state_idx = %d" % (r2_state_prime_tup, r2_state_prime_idx), 'cyan', attrs = ['bold']))

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

        logging.debug("%s", colored("************************************* End of Trial %d ****************************************************" % (i), 'white', attrs = ['bold']))

    r1_feature_expectation = r1_feature_expectation/n_experiments
    r2_feature_expectation = r2_feature_expectation/n_experiments

    return r1_feature_expectation/np.linalg.norm(r1_feature_expectation, ord = 1), r2_feature_expectation/np.linalg.norm(r2_feature_expectation, ord = 1)

def compute_mu_bar_curr(mu_e, mu_bar_prev, mu_curr):
    """Function to compute mu_bar_current using the projection forumula from Abbeel and Ng's paper page 4
    """
    x = mu_curr - mu_bar_prev
    y = mu_e - mu_bar_prev
    mu_bar_curr = mu_bar_prev + (np.dot(x.T, y)/np.dot(x.T, x)) * x

    return mu_bar_curr

def softmax(w, t = 1.0):
    """Function to calculate the softmax of a matrix, normalizing each row for probability
    """
    e = np.exp(w / t)
    dist = e/e.sum(axis=1).reshape((len(e), 1))
    return dist

def team_q_learning(r1_state_action_dist, r1_reward, r2_state_action_dist, r2_reward, n_episodes = 10):
    """Function that runs the q learning algorithm for both the agents and returns the action_distribution (after softmaxing it)
    """
    logging.debug("%s", colored("                                        TEAM_Q_LEARNING                     ", 'white', attrs = ['bold']))
    r1_q = np.zeros((n_states, ts.n_action_vars))
    r2_q = np.zeros((n_states, ts.n_action_vars))

    # for actions that cannot be taken in particular states, set q value to be -inf so that, that action will never be chosen
    r1_q[state_action_space == -np.inf] = -np.inf
    r2_q[state_action_space == -np.inf] = -np.inf

    for episode in range(n_episodes):
        logging.debug("%s", colored("************************************* Episode %d ****************************************************" % (episode), 'white', attrs = ['bold']))
        start_state = random.choice(task_start_state_set)
        r1_state_idx = task_states_list.index(start_state)
        r2_state_idx = r1_state_idx
        r1_state_tup = start_state
        r2_state_tup = start_state
        step = 1

        while True:
            logging.debug("%s", colored("************************************* Step %d ****************************************************" % (step), 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_action_dist = %s" % (r1_state_action_dist[r1_state_idx]), 'red', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_action_dist = %s" % (r1_state_action_dist[r1_state_idx]), 'cyan', attrs = ['bold']))

            r1_action = select_random_action(r1_state_action_dist[r1_state_idx])
            r2_action = select_random_action(r2_state_action_dist[r2_state_idx])
            r1_action_idx = ts.task_actions_expl[r1_action][0]
            r2_action_idx = ts.task_actions_expl[r2_action][0]


            logging.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ BEFORE ACTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_tup = %s, state_idx = %d" % (r1_state_tup, r1_state_idx), 'red', attrs = ['bold']))
            logging.debug("%s", colored("r2_state_tup = %s, state_idx = %d" % (r2_state_tup, r2_state_idx), 'cyan', attrs = ['bold']))
            logging.debug("%s", colored("r1_action = %s, action_idx = %d" % (r1_action, r1_action_idx), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_action = %s, action_idx = %d" % (r2_action, r2_action_idx), 'cyan', attrs = ['bold']))

            if r1_action == 'X' and r2_action == 'X':
                logging.debug("%s", colored("************************************* End of Step %d ****************************************************" % (step), 'white', attrs = ['bold']))
                break

            r1_state_prime_tup, r2_state_prime_tup = sf.simulate_next_state(r1_action, r1_state_tup, r2_state_tup) # first agent acting

            logging.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ After 1st Agent Action @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_prime_tup = %s" % str(r1_state_prime_tup), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_state_prime_tup = %s" % str(r2_state_prime_tup), 'cyan', attrs = ['bold']))

            r2_state_prime_tup, r1_state_prime_tup = sf.simulate_next_state(r2_action, r2_state_prime_tup, r1_state_prime_tup) # second agent acting
            r1_state_prime_idx = task_states_list.index(r1_state_prime_tup)
            r2_state_prime_idx = task_states_list.index(r2_state_prime_tup)

            logging.debug("%s\n", colored("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ After 2nd Agent Action @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_state_prime_tup = %s, state_idx = %d" % (r1_state_prime_tup, r1_state_prime_idx), 'red', attrs = ['bold']))
            logging.debug("%s", colored("r2_state_prime_tup = %s, state_idx = %d" % (r2_state_prime_tup, r2_state_prime_idx), 'cyan', attrs = ['bold']))

            # get max action index for both agents
            r1_max_action_idx = r1_q[r1_state_prime_idx].argmax()
            r2_max_action_idx = r2_q[r2_state_prime_idx].argmax()

            logging.debug("%s", colored("r1_max_action = %s, action_idx = %d" % (ts.task_actions_index[r1_max_action_idx], r1_max_action_idx), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_max_action = %s, action_idx = %d" % (ts.task_actions_index[r2_max_action_idx], r2_max_action_idx), 'cyan', attrs = ['bold']))

            logging.debug("%s\n", colored("################################## Q Value Before Update #####################################", 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_q[%d][%d] = %f" % (r1_state_idx, r1_action_idx, r1_q[r1_state_idx][r1_action_idx]), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_q[%d][%d] = %f" % (r2_state_idx, r2_action_idx, r2_q[r2_state_idx][r2_action_idx]), 'cyan', attrs = ['bold']))

            r1_q[r1_state_idx][r1_action_idx] = r1_q[r1_state_idx][r1_action_idx] + alpha * (r1_reward[r1_state_idx][r1_action_idx] +
                    gamma * r1_q[r1_state_prime_idx][r1_max_action_idx] - r1_q[r1_state_idx][r1_action_idx])
            r2_q[r2_state_idx][r2_action_idx] = r2_q[r2_state_idx][r2_action_idx] + alpha * (r2_reward[r2_state_idx][r2_action_idx] +
                    gamma * r2_q[r2_state_prime_idx][r2_max_action_idx] - r2_q[r2_state_idx][r2_action_idx])

            logging.debug("%s\n", colored("################################## Q Value After Update #####################################", 'white', attrs = ['bold']))
            logging.debug("%s", colored("r1_q[%d][%d] = %f" % (r1_state_idx, r1_action_idx, r1_q[r1_state_idx][r1_action_idx]), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_q[%d][%d] = %f" % (r2_state_idx, r2_action_idx, r2_q[r2_state_idx][r2_action_idx]), 'cyan', attrs = ['bold']))

            # update current states to new states
            r1_state_tup = r1_state_prime_tup
            r2_state_tup = r2_state_prime_tup

            # update the state indices for both agents
            r1_state_idx = task_states_list.index(r1_state_tup)
            r2_state_idx = task_states_list.index(r2_state_tup)

            logging.debug("%s", colored("************************************* End of Step %d ****************************************************" % (step), 'white', attrs = ['bold']))
            step = step + 1
            #user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
            #if user_input.upper() == 'Q':
               #break;

        logging.debug("%s", colored("************************************* End of Episode %d ****************************************************" % (episode), 'white', attrs = ['bold']))

    return softmax(r1_q), softmax(r2_q)

def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    mu_e_normalized = expert_feature_expectation/np.linalg.norm(expert_feature_expectation, ord = 1)

    i = 1
    while True:
        if i == 1:
            r1_state_action_dist = compute_random_state_action_distribution()
            r2_state_action_dist = compute_random_state_action_distribution()
        else:
            r1_state_action_dist, r2_state_action_dist = team_q_learning(r1_state_action_dist, r1_reward, r2_state_action_dist, r2_reward)

        mu_curr_r1, mu_curr_r2 = compute_normalized_feature_expectation(r1_state_action_dist, r2_state_action_dist)

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
            logging.info("%s", colored("***************** Iteration %d ***********************" % (i), 'white', attrs = ['bold']))
            logging.info("%s\n", colored("mu_e_normalized = %s" % (mu_e_normalized), 'green', attrs = ['bold']))

            logging.info("%s", colored("mu_curr_r1 = %s" % (mu_curr_r1), 'red', attrs = ['bold']))
            logging.info("%s\n", colored("mu_curr_r2 = %s" % (mu_curr_r2), 'cyan', attrs = ['bold']))

            logging.info("%s", colored("mu_bar_curr_r1 = %s" % (mu_bar_curr_r1), 'red', attrs = ['bold']))
            logging.info("%s\n", colored("mu_bar_curr_r2 = %s" % (mu_bar_curr_r2), 'cyan', attrs = ['bold']))

            logging.info("%s", colored("r1_w 1-norm = %s" % (np.linalg.norm(r1_w, ord = 1)), 'red', attrs = ['bold']))
            logging.info("%s\n", colored("r2_w 1-norm = %s" % (np.linalg.norm(r2_w, ord = 1)), 'cyan', attrs = ['bold']))

            logging.info("%s", colored("r1_t = %s" % (r1_t), 'red', attrs = ['bold']))
            logging.info("%s", colored("r2_t = %s" % (r2_t), 'cyan', attrs = ['bold']))

            logging.info("%s", colored("***************** End of Iteration %d ***********************" % (i), 'white', attrs = ['bold']))
            user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
            if user_input.upper() == 'Q':
               break;

        i = i + 1
        mu_bar_prev_r1 = mu_bar_curr_r1
        mu_bar_prev_r2 = mu_bar_curr_r2

if __name__=='__main__':
    main()

