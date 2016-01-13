#!/usr/bin/env python

import sys
import logging
import cPickle as pickle
import random
import pprint
import numpy as np

from termcolor import colored

import taskSetup as ts
import simulationFunctions as sf

task_states, task_start_states, task_state_action_map, phi_matrix = ts.load_state_data()

def init_random_policy(is_policy = True):
    random_policy = dict()

    for task_state, state_actions in task_state_action_map.items():
        actions = dict()
        for action in state_actions:
            actions[action] = random.random()
        if is_policy:
            actions = {k: float(v)/ total for total in (sum(actions.values()),) for k, v in actions.items()}
        else:
            actions = {k: 2*float(v)-1 for k, v in actions.items()}
        random_policy[task_state] = actions
    return random_policy

def simulate_random_poliy():
    print "Total number of actions by agents using random policy is %d" % sf.run_simulation(init_random_policy(), init_random_policy(), random.choice(tuple(task_start_states)))

def get_mu(pi_1, pi_2, n_trials):
    mu_r1 = np.zeros(ts.n_state_vars + len(ts.task_actions))
    mu_r2 = np.zeros(ts.n_state_vars + len(ts.task_actions))
    for i in range(n_trials):
        start_state = random.choice(tuple(task_start_states))
        state_r1 = start_state
        state_r2 = start_state
        while True:
            action_r1 = sf.random_select_action(pi_1[state_r1])
            action_r2 = sf.random_select_action(pi_2[state_r2])
            if action_r1 == 'X' and action_r2 == 'X':
                break
            state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
            state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
            mu_r1 = mu_r1 + ts.get_phi(list(state_r1), action_r1)
            mu_r2 = mu_r2 + ts.get_phi(list(state_r2), action_r2)
    mu_r1 = mu_r1/n_trials
    mu_r2 = mu_r2/n_trials
    # normalizing feature expection to bind the first norm of rewards and w within 1
    return mu_r1/np.linalg.norm(mu_r1), mu_r2/np.linalg.norm(mu_r2)

def q_learning(pi_r1, reward_r1, pi_r2, reward_r2):
    q_r1 = init_random_policy(False)
    q_r2 = init_random_policy(False)
    states = {v: k for k, v in task_states.items()}
    actions = ts.task_actions.keys()
    n_episodes = 1000
    alpha = 0.2
    gamma = 1.0
    for i in range(n_episodes):
        start_state = random.choice(tuple(task_start_states))
        state_r1 = start_state
        state_r2 = start_state
        while True:
            action_r1 = sf.random_select_action(pi_r1[state_r1])
            action_r2 = sf.random_select_action(pi_r2[state_r2])
            if action_r1 == 'X' and action_r2 == 'X':
                break
            state_r1_prime, state_r2_prime = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
            state_r2_prime, state_r1_prime = sf.simulate_next_state(action_r2, state_r2_prime, state_r1_prime) # second agent acting

            # Update q for first agent
            q_r1[state_r1][action_r1] = q_r1[state_r1][action_r1] + alpha * (reward_r1[states[state_r1]][actions.index(action_r1)] + gamma * max(q_r1[state_r1_prime].values()) - q_r1[state_r1][action_r1])
            # Update q for second agent
            q_r2[state_r2][action_r2] = q_r2[state_r2][action_r2] + alpha * (reward_r2[states[state_r2]][actions.index(action_r2)] + gamma * max(q_r2[state_r2_prime].values()) - q_r2[state_r2][action_r2])

            state_r1 = state_r1_prime
            state_r2 = state_r2_prime
    #print pprint.pprint(q_r1)
    #print pprint.pprint(q_r2)


def main():
    _, _, mu_e, n_trials = ts.read_task_data()

    logging.basicConfig(level=logging.WARN, format='%(asctime)s-%(levelname)s: %(message)s')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    #simulate_random_poliy(task_start_states, task_state_action_map)

    # normalizing feature expection to bind the first norm of rewards and w within 1
    print mu_e
    mu_e = mu_e/np.linalg.norm(mu_e)
    # first iteration
    pi_r1 = init_random_policy()
    pi_r2 = init_random_policy()
    mu_curr_r1, mu_curr_r2 = get_mu(pi_r1, pi_r2, n_trials)
    i = 1

    mu_bar_curr_r1 = mu_curr_r1
    mu_bar_curr_r2 = mu_curr_r2
    w_r1 = (mu_e - mu_bar_curr_r1)
    w_r2 = (mu_e - mu_bar_curr_r2)
    t_r1 = np.linalg.norm(w_r1)
    t_r2 = np.linalg.norm(w_r2)
    reward_r1 = np.reshape(np.dot(phi_matrix, w_r1), (len(task_states), len(ts.task_actions)))
    reward_r2 = np.reshape(np.dot(phi_matrix, w_r2), (len(task_states), len(ts.task_actions)))

    ######## Use reward_table on mdp to get new policy  ###########
    pi_r1 = init_random_policy() # this should come from mdp solution
    pi_r2 = init_random_policy() # this should come from mdp solution
    mu_curr_r1, mu_curr_r2 = get_mu(pi_r1, pi_r2, n_trials)
    mu_bar_prev_r1 = mu_bar_curr_r1
    mu_bar_prev_r2 = mu_bar_curr_r2
    q_learning(pi_r1, reward_r1, pi_r2, reward_r2)
    # while max(t_r1, t_r2) > 0.1:
        # print "Iteration: ", i
        # print "mu_bar_prev_r1 = ", mu_bar_prev_r1
        # print "mu_bar_prev_r2 = ", mu_bar_prev_r2
        # x = mu_curr_r1 - mu_bar_prev_r1
        # y = mu_e - mu_bar_prev_r1
        # mu_bar_curr_r1 = mu_bar_prev_r1 + (np.dot(x.T, y)/np.dot(x.T, x)) * x
        # x = mu_curr_r2 - mu_bar_prev_r2
        # y = mu_e - mu_bar_prev_r2
        # mu_bar_curr_r2 = mu_bar_prev_r2 + (np.dot(x.T, y)/np.dot(x.T, x)) * x
        # rstate_idx = random.randrange(0, len(task_states))
        # print "mu_bar_curr_r1 = ", mu_bar_curr_r1
        # print "mu_bar_curr_r2 = ", mu_bar_curr_r2
        # print "reward_r1[", rstate_idx, "] = ", reward_r1[rstate_idx]
        # print "reward_r2[", rstate_idx, "] = ", reward_r2[rstate_idx]
        # print "t_r1 = ", np.round(t_r1, 3)
        # print "t_r2 = ", np.round(t_r2, 3)
        # w_r1 = (mu_e - mu_bar_curr_r1)
        # w_r2 = (mu_e - mu_bar_curr_r2)
        # t_r1 = np.linalg.norm(w_r1)
        # t_r2 = np.linalg.norm(w_r2)
        # reward_r1 = np.reshape(np.dot(phi_matrix, w_r1), (len(task_states), len(ts.task_actions)))
        # reward_r2 = np.reshape(np.dot(phi_matrix, w_r2), (len(task_states), len(ts.task_actions)))
        # i = i + 1

        # ##### Use reward_table on mdp to get new policy  ###########
        # pi_r1 = init_random_policy() # this should come from mdp solution
        # pi_r2 = init_random_policy() # this should come from mdp solution
        # mu_curr_r1, mu_curr_r2 = get_mu(pi_r1, pi_r2, n_trials)
        # mu_bar_prev_r1 = mu_bar_curr_r1
        # mu_bar_prev_r2 = mu_bar_curr_r2
        # print "**********************************************************"
        # user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        # if user_input.upper() == 'Q':
            # break;
        # print "**********************************************************"

if __name__=='__main__':
    main()

