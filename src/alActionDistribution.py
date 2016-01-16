#!/usr/bin/env python

import sys
import logging
import random
import pprint
import numpy as np

from termcolor import colored
import taskSetup as ts
import simulationFunctions as sf

# set logging level
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s: %(message)s')

# load task params from pickle file
task_params = ts.load_task_parameters()
task_states_list = task_params[ts.TaskParams.task_states_list]
task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]
feature_matrix = task_params[ts.TaskParams.feature_matrix]
expert_visited_states_set = task_params[ts.TaskParams.expert_visited_states_set]
expert_state_action_dict = task_params[ts.TaskParams.expert_state_action_dict]
expert_feature_expectation = task_params[ts.TaskParams.expert_feature_expectation]
n_experiments = task_params[ts.TaskParams.n_experiments]
time_per_step = task_params[ts.TaskParams.time_per_step]

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
alpha = 0.2
gamma = 1.0

def compute_random_state_action_distribution():
    """Function to compute a random state action distribution
    """
    random_state_action_dist = np.random.rand(n_states, ts.n_action_vars)
    random_state_action_dist[state_action_space == -np.inf] = 0
    random_state_action_dist = random_state_action_dist / random_state_action_dist.sum(axis=1).reshape((len(random_state_action_dist), 1)) # normalize each row

    return random_state_action_dist

def extract_policy(q_state_action_matrix):
    """Function to extract the policy from q_value matrix using argmax
    """
    policy = dict()
    for state_idx, task_state_tup in enumerate(task_states_list):
        action_idx = np.argmax(q_state_action_matrix[state_idx])
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
    r1_feature_expectation = np.zeros(ts.n_state_vars + ts.n_action_vars)
    r2_feature_expectation = np.zeros(ts.n_state_vars + ts.n_action_vars)

    for i in range(n_experiments):
        start_state = random.choice(task_start_state_set)
        r1_state_idx = task_states_list.index(start_state)
        r2_state_idx = r1_state_idx
        r1_state_tup = start_state
        r2_state_tup = start_state
        step = 1

        while True:
            print "************************************* STEP IN COMPUTE MU_E", step, "****************************************************"
            print "r1_state_action_dist = ", r1_state_action_dist[r1_state_idx]
            r1_action = select_random_action(r1_state_action_dist[r1_state_idx])
            print "r2_state_action_dist = ", r2_state_action_dist[r1_state_idx]
            r2_action = select_random_action(r2_state_action_dist[r2_state_idx])
            r1_action_idx = ts.task_actions_expl[r1_action][0]
            r2_action_idx = ts.task_actions_expl[r2_action][0]

            if r1_action == 'X' and r2_action == 'X':
                print "************************************* END OF STEP IN COMPUTE MU_E", step, "****************************************************"
                break

            r1_state_prime_tup, r2_state_prime_tup = sf.simulate_next_state(r1_action, r1_state_tup, r2_state_tup) # first agent acting
            r2_state_prime_tup, r1_state_prime_tup = sf.simulate_next_state(r2_action, r2_state_prime_tup, r1_state_prime_tup) # second agent acting

            r1_feature_expectation = r1_feature_expectation + feature_matrix[get_feature_idx(r1_state_idx, r1_action_idx)]
            r2_feature_expectation = r2_feature_expectation + feature_matrix[get_feature_idx(r2_state_idx, r2_action_idx)]

            # update current states to new states
            r1_state_tup = r1_state_prime_tup
            r2_state_tup = r2_state_prime_tup

            # update the state indices for both agents
            r1_state_idx = task_states_list.index(r1_state_tup)
            r2_state_idx = task_states_list.index(r2_state_tup)

            #print "************************************* END OF STEP IN COMPUTE MU_E", step, "****************************************************"
            step = step + 1

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

def team_q_learning(r1_state_action_dist, r1_reward, r2_state_action_dist, r2_reward, n_episodes = 1):
    r1_q = np.zeros((n_states, ts.n_action_vars))
    r2_q = np.zeros((n_states, ts.n_action_vars))

    for episode in range(n_episodes):
        start_state = random.choice(task_start_state_set)
        r1_state_idx = task_states_list.index(start_state)
        r2_state_idx = r1_state_idx
        r1_state_tup = start_state
        r2_state_tup = start_state
        step = 1

        while True:
            #print "************************************* STEP ", step, "****************************************************"
            #print "r1_state_action_dist = ", r1_state_action_dist[r1_state_idx]
            r1_action = select_random_action(r1_state_action_dist[r1_state_idx])
            #print "r2_state_action_dist = ", r2_state_action_dist[r1_state_idx]
            r2_action = select_random_action(r2_state_action_dist[r2_state_idx])
            r1_action_idx = ts.task_actions_expl[r1_action][0]
            r2_action_idx = ts.task_actions_expl[r2_action][0]


            #print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ BEFORE ACTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            #print
            logging.debug("%s", colored("r1_state_tup = %s, state_idx = %d" % (r1_state_tup, r1_state_idx), 'red', attrs = ['bold']))
            logging.debug("%s", colored("r2_state_tup = %s, state_idx = %d" % (r2_state_tup, r2_state_idx), 'cyan', attrs = ['bold']))
            logging.debug("%s", colored("r1_action = %s, action_idx = %d" % (r1_action, r1_action_idx), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_action = %s, action_idx = %d" % (r2_action, r2_action_idx), 'cyan', attrs = ['bold']))

            if r1_action == 'X' and r2_action == 'X':
                #print "************************************* END OF STEP ", step, "****************************************************"
                break


            r1_state_prime_tup, r2_state_prime_tup = sf.simulate_next_state(r1_action, r1_state_tup, r2_state_tup) # first agent acting

            #print "@@@@@@@@@@@@@@@@@@@@@ AFTER 1st AGENT ACTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            #print
            logging.debug("%s", colored("r1_state_prime_tup = %s" % str(r1_state_prime_tup), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_state_prime_tup = %s" % str(r2_state_prime_tup), 'cyan', attrs = ['bold']))

            r2_state_prime_tup, r1_state_prime_tup = sf.simulate_next_state(r2_action, r2_state_prime_tup, r1_state_prime_tup) # second agent acting
            r1_state_prime_idx = task_states_list.index(r1_state_prime_tup)
            r2_state_prime_idx = task_states_list.index(r2_state_prime_tup)

            #print "@@@@@@@@@@@@@@@@@@@@@ AFTER 2nd AGENT ACTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            #print
            logging.debug("%s", colored("r1_state_prime_tup = %s, state_idx = %d" % (r1_state_prime_tup, r1_state_prime_idx), 'red', attrs = ['bold']))
            logging.debug("%s", colored("r2_state_prime_tup = %s, state_idx = %d" % (r2_state_prime_tup, r2_state_prime_idx), 'cyan', attrs = ['bold']))

            # get max action index for both agents
            r1_max_action_idx = r1_q[r1_state_prime_idx].argmax()
            r2_max_action_idx = r2_q[r2_state_prime_idx].argmax()

            logging.debug("%s", colored("r1_max_action = %s, action_idx = %d" % (ts.task_actions_index[r1_max_action_idx], r1_max_action_idx), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_max_action = %s, action_idx = %d" % (ts.task_actions_index[r2_max_action_idx], r2_max_action_idx), 'cyan', attrs = ['bold']))

            #print "################################## Q BEFORE UPDATE #####################################"
            #print
            logging.debug("%s", colored("r1_q[%d][%d] = %f" % (r1_state_idx, r1_action_idx, r1_q[r1_state_idx][r1_action_idx]), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_q[%d][%d] = %f" % (r2_state_idx, r2_action_idx, r2_q[r2_state_idx][r2_action_idx]), 'cyan', attrs = ['bold']))

            r1_q[r1_state_idx][r1_action_idx] = r1_q[r1_state_idx][r1_action_idx] + alpha * (r1_reward[r1_state_idx][r1_action_idx] +
                    gamma * r1_q[r1_state_prime_idx][r1_max_action_idx] - r1_q[r1_state_idx][r1_action_idx])
            r2_q[r2_state_idx][r2_action_idx] = r2_q[r2_state_idx][r2_action_idx] + alpha * (r2_reward[r2_state_idx][r2_action_idx] +
                    gamma * r2_q[r2_state_prime_idx][r2_max_action_idx] - r2_q[r2_state_idx][r2_action_idx])

            #print "################################## Q AFTER UPDATE #####################################"
            #print
            logging.debug("%s", colored("r1_q[%d][%d] = %f" % (r1_state_idx, r1_action_idx, r1_q[r1_state_idx][r1_action_idx]), 'red', attrs = ['bold']))
            logging.debug("%s\n", colored("r2_q[%d][%d] = %f" % (r2_state_idx, r2_action_idx, r2_q[r2_state_idx][r2_action_idx]), 'cyan', attrs = ['bold']))

            # update current states to new states
            r1_state_tup = r1_state_prime_tup
            r2_state_tup = r2_state_prime_tup

            # update the state indices for both agents
            r1_state_idx = task_states_list.index(r1_state_tup)
            r2_state_idx = task_states_list.index(r2_state_tup)

            #print "************************************* END OF STEP ", step, "****************************************************"
            step = step + 1
            #user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
            #if user_input.upper() == 'Q':
               #break;

    return softmax(r1_q), softmax(r2_q)

def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    mu_e_normalized = expert_feature_expectation/np.linalg.norm(expert_feature_expectation, ord = 1)

    #dist = compute_random_state_action_distribution()
    #action = select_random_action(dist[np.random.randint(0, n_states)])
    #print action
    #print dist

    i = 1
    while True:
        print "Iteration: ", i
        if i == 1:
            r1_state_action_dist = compute_random_state_action_distribution()
            r2_state_action_dist = compute_random_state_action_distribution()
        else:
            r1_state_action_dist, r2_state_action_dist = team_q_learning(r1_state_action_dist, r1_reward, r2_state_action_dist, r2_reward)

        #print r1_state_action_dist
        #print "r1 = ", np.sum(r1_state_action_dist)
        #print "r2 = ", np.sum(r2_state_action_dist)
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


        #if i != 1:
            #print "mu_curr_r1 = ", mu_curr_r1, "\n"
            #print "mu_bar_curr_r1 = ", mu_bar_curr_r1, "\n"
            #print "mu_curr_r2 = ", mu_curr_r2, "\n"
            #print "mu_bar_curr_r2 = ", mu_bar_curr_r2, "\n"
            #print "r1_w = ", np.linalg.norm(r1_w, ord = 1)
            #print "r2_w = ", np.linalg.norm(r2_w, ord = 1)
            #print
            #print "r1_t = ", r1_t
            #print "r2_t = ", r2_t
        print "**********************************************************"
        user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        if user_input.upper() == 'Q':
           break;
        print "**********************************************************"

        i = i + 1
        mu_bar_prev_r1 = mu_bar_curr_r1
        mu_bar_prev_r2 = mu_bar_curr_r2

if __name__=='__main__':
    main()
