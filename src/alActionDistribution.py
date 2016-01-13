#!/usr/bin/env python

import logging
import random
import pprint
import numpy as np

import taskSetup as ts
import simulationFunctions as sf

def compute_random_state_action_distribution_dict(task_state_action_dict):
    """Function to compute a random distribution for actions for each task state
    """
    random_state_action_distribution_dict = dict()
    for task_state_tup, actions_dict in task_state_action_dict.items():
        random_actions = dict()
        for action in actions_dict:
            random_actions[action] = random.random()
        random_actions = {k: float(v)/total for total in (sum(random_actions.values()),) for k, v in random_actions.items()}
        random_state_action_distribution_dict[task_state_tup] = random_actions

    return random_state_action_distribution_dict

def simulate_random_state_action_distribution_dict():
    """Function to simulate a random action distribution for both the agents
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')
    task_states_set = task_params[ts.TaskParams.task_states_set]
    task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
    task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]

    r1_state_action_distribution_dict = compute_random_state_action_distribution_dict(task_state_action_dict)
    r2_state_action_distribution_dict = compute_random_state_action_distribution_dict(task_state_action_dict)
    print "Total number of actions by agents using random policy is %d" % sf.run_simulation(r1_state_action_distribution_dict, r2_state_action_distribution_dict, random.choice(tuple(task_start_state_set)))

def compute_normalized_feature_expectation(task_start_state_set, r1_state_action_distribution_dict, r2_state_action_distribution_dict, n_trials):
    """Function to compute the feature expectation of the agents by running the simulation for n_trials. The feature expectations are normalized to bind them within 1
    """
    r1_feature_expectation = np.zeros(ts.n_state_vars + ts.n_action_vars)
    r2_feature_expectation = np.zeros(ts.n_state_vars + ts.n_action_vars)

    for trial in range(n_trials):
        start_state = random.choice(tuple(task_start_state_set))
        r1_state_tup = start_state
        r2_state_tup = start_state
        while True:
            r1_action = sf.select_random_action(r1_state_action_distribution_dict[r1_state_tup])
            r2_action = sf.select_random_action(r2_state_action_distribution_dict[r2_state_tup])

            if r1_action == 'X' and r2_action == 'X':
                break

            r1_state_tup, r2_state_tup = sf.simulate_next_state(r1_action, r1_state_tup, r2_state_tup) # first agent acting
            r2_state_tup, r1_state_tup = sf.simulate_next_state(r2_action, r2_state_tup, r1_state_tup) # second agent acting

            # compute feature expectations of the agents
            r1_feature_expectation = r1_feature_expectation + ts.get_feature_vector(r1_state_tup, r1_action)
            r2_feature_expectation = r2_feature_expectation + ts.get_feature_vector(r2_state_tup, r2_action)

    r1_feature_expectation = r1_feature_expectation/n_trials
    r2_feature_expectation = r2_feature_expectation/n_trials

    # normalizing feature expection to bind the first norm of rewards and w within 1
    return r1_feature_expectation/np.linalg.norm(r1_feature_expectation), r2_feature_expectation/np.linalg.norm(r2_feature_expectation)

def compute_mu_bar_curr(mu_e, mu_bar_prev, mu_curr):
    x = mu_curr - mu_bar_prev
    y = mu_e - mu_bar_prev
    mu_bar_curr = mu_bar_prev + (np.dot(x.T, y)/np.dot(x.T, x)) * x
    return mu_bar_curr

def numpyfy_state_action_dict(state_action_dict):
    state_action_narray = np.empty((0, ts.n_action_vars))
    for task_state_tup, action_dict in state_action_dict.items():
        # get the indices of the valid actions for the task_state_tup from state_action_dict
        action_idx = [ts.task_actions_dict[_][0] for _ in action_dict.keys()]

        # create a row (vector) for the current task_state_tup
        current_task_vector = np.zeros(ts.n_action_vars)
        np.put(current_task_vector, action_idx, action_dict.values())

        # add the row to the matrix
        state_action_narray = np.vstack((state_action_narray, current_task_vector))

        #print task_state_tup
        #print action_dict
        #print current_task_vector

    return state_action_narray

def q_learning():
    pass

def main():
    logging.basicConfig(level=logging.WARN, format='%(asctime)s-%(levelname)s: %(message)s')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    task_params = ts.load_task_parameters()
    task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
    task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]
    feature_matrix = task_params[ts.TaskParams.feature_matrix]
    expert_feature_expectation = task_params[ts.TaskParams.expert_feature_expectation]
    task_states_narray = task_params[ts.TaskParams.task_states_narray]
    task_state_action_narray = task_params[ts.TaskParams.task_state_action_narray]
    n_trials = task_params[ts.TaskParams.n_trials]

    # normalizing expert feature expection to bind the first norm of rewards and w within 1
    mu_e_normalized = expert_feature_expectation/np.linalg.norm(expert_feature_expectation)

    # first iteration
    i = 1
    r1_state_action_distribution_dict = compute_random_state_action_distribution_dict(task_state_action_dict)
    r2_state_action_distribution_dict = compute_random_state_action_distribution_dict(task_state_action_dict)
    mu_curr_r1, mu_curr_r2 = compute_normalized_feature_expectation(task_start_state_set, r1_state_action_distribution_dict, r2_state_action_distribution_dict, n_trials)
    mu_bar_curr_r1 = mu_curr_r1
    mu_bar_curr_r2 = mu_curr_r2

    w_r1 = (mu_e_normalized - mu_bar_curr_r1)
    w_r2 = (mu_e_normalized - mu_bar_curr_r2)
    t_r1 = np.linalg.norm(w_r1)
    t_r2 = np.linalg.norm(w_r2)
    reward_r1 = np.reshape(np.dot(feature_matrix, w_r1), (len(task_states_narray), ts.n_action_vars))
    reward_r2 = np.reshape(np.dot(feature_matrix, w_r2), (len(task_states_narray), ts.n_action_vars))

    mu_bar_prev_r1 = mu_bar_curr_r1
    mu_bar_prev_r2 = mu_bar_curr_r2

    r1_state_action_dist = numpyfy_state_action_dict(r1_state_action_distribution_dict)
    r2_state_action_dist = numpyfy_state_action_dict(r2_state_action_distribution_dict)

    print r1_state_action_dist.shape, r1_state_action_dist.size
    #q_learning(pi_r1, reward_r1, pi_r2, reward_r2)

    #while True:
        #print "Iteration: ", i
        #print "mu_bar_prev_r1 = ", mu_bar_prev_r1
        #print "mu_bar_prev_r2 = ", mu_bar_prev_r2

        ###################### Use computed reward on qlearning algorithm to compute new state_action_distribution_dict ##########################
        #r1_state_action_distribution_dict = compute_random_state_action_distribution_dict(task_state_action_dict) # this should come from mdp solution
        #r2_state_action_distribution_dict = compute_random_state_action_distribution_dict(task_state_action_dict) # this should come from mdp solution
        #mu_curr_r1, mu_curr_r2 = compute_normalized_feature_expectation(task_start_state_set, r1_state_action_distribution_dict, r2_state_action_distribution_dict, n_trials)
        #mu_bar_curr_r1 = compute_mu_bar_curr(mu_e_normalized, mu_bar_prev_r1, mu_curr_r1)
        #mu_bar_curr_r2 = compute_mu_bar_curr(mu_e_normalized, mu_bar_prev_r2, mu_curr_r2)

        #rstate_idx = random.randrange(0, len(task_states_narray))
        #print "mu_bar_curr_r1 = ", mu_bar_curr_r1
        #print "mu_bar_curr_r2 = ", mu_bar_curr_r2
        #print "reward_r1[", rstate_idx, "] = ", reward_r1[rstate_idx]
        #print "reward_r2[", rstate_idx, "] = ", reward_r2[rstate_idx]
        #print "t_r1 = ", np.round(t_r1, 3)
        #print "t_r2 = ", np.round(t_r2, 3)

        ## update the weights
        #w_r1 = (mu_e_normalized - mu_bar_curr_r1)
        #w_r2 = (mu_e_normalized - mu_bar_curr_r2)
        #t_r1 = np.linalg.norm(w_r1)
        #t_r2 = np.linalg.norm(w_r2)
        #reward_r1 = np.reshape(np.dot(feature_matrix, w_r1), (len(task_states_narray), ts.n_action_vars))
        #reward_r2 = np.reshape(np.dot(feature_matrix, w_r2), (len(task_states_narray), ts.n_action_vars))

        #i = i + 1
        #mu_bar_prev_r1 = mu_bar_curr_r1
        #mu_bar_prev_r2 = mu_bar_curr_r2

        #print "**********************************************************"
        #user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        #if user_input.upper() == 'Q':
           #break;
        #print "**********************************************************"


#def q_learning(pi_r1, reward_r1, pi_r2, reward_r2):
    #q_r1 = init_random_policy(False)
    #q_r2 = init_random_policy(False)
    #states = {v: k for k, v in task_states.items()}
    #actions = ts.task_actions.keys()
    #n_episodes = 1000
    #alpha = 0.2
    #gamma = 1.0
    #for i in range(n_episodes):
        #start_state = random.choice(tuple(task_start_states))
        #state_r1 = start_state
        #state_r2 = start_state
        #while True:
            #action_r1 = sf.random_select_action(pi_r1[state_r1])
            #action_r2 = sf.random_select_action(pi_r2[state_r2])
            #if action_r1 == 'X' and action_r2 == 'X':
                #break
            #state_r1_prime, state_r2_prime = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
            #state_r2_prime, state_r1_prime = sf.simulate_next_state(action_r2, state_r2_prime, state_r1_prime) # second agent acting

            ## Update q for first agent
            #q_r1[state_r1][action_r1] = q_r1[state_r1][action_r1] + alpha * (reward_r1[states[state_r1]][actions.index(action_r1)] + gamma * max(q_r1[state_r1_prime].values()) - q_r1[state_r1][action_r1])
            ## Update q for second agent
            #q_r2[state_r2][action_r2] = q_r2[state_r2][action_r2] + alpha * (reward_r2[states[state_r2]][actions.index(action_r2)] + gamma * max(q_r2[state_r2_prime].values()) - q_r2[state_r2][action_r2])

            #state_r1 = state_r1_prime
            #state_r2 = state_r2_prime
    ##print pprint.pprint(q_r1)
    ##print pprint.pprint(q_r2)

if __name__=='__main__':
    main()

