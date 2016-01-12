#!/usr/bin/env python

import sys
import logging
import cPickle as pickle
import random
import numpy as np

from termcolor import colored

import taskSetup as ts
import simulationFunctions as sf

task_states, task_start_states, task_state_action_map, phi_matrix = ts.load_state_data()

def init_random_policy(task_state_action_map):
    random_policy = dict()

    for task_state, state_actions in task_state_action_map.items():
        actions = dict()
        for action in state_actions:
            actions[action] = random.random()
        actions = {k: float(v)/ total for total in (sum(actions.values()),) for k, v in actions.items()}
        random_policy[task_state] = actions
    return random_policy

def simulate_random_poliy(task_start_states, task_state_action_map):
    print "Total number of actions by agents using random policy is %d" % sf.run_simulation(init_random_policy(task_state_action_map), init_random_policy(task_state_action_map), random.choice(tuple(task_start_states)))

def get_mu(pi_1, pi_2, n_trials):
    mu = np.zeros(ts.n_state_vars + len(ts.task_actions))
    for i in range(n_trials):
        start_state = random.choice(tuple(task_start_states))
        state_r1 = start_state
        state_r2 = start_state
        while True:
            action_r1 = sf.softmax_select_action(pi_1[state_r1])
            action_r2 = sf.softmax_select_action(pi_2[state_r2])
            if action_r1 == 'X' and action_r2 == 'X':
                break
            state_r1, state_r2 = sf.simulate_next_state(action_r1, state_r1, state_r2) # first agent acting
            state_r2, state_r1 = sf.simulate_next_state(action_r2, state_r2, state_r1) # second agent acting
            mu = mu + ts.get_phi(list(state_r1), action_r1)
    mu = mu/n_trials
    # normalizing feature expection to bind the first norm of rewards and w within 1
    return mu/np.linalg.norm(mu)

def main():
    _, _, mu_e, n_trials = ts.read_task_data()

    logging.basicConfig(level=logging.WARN, format='%(asctime)s-%(levelname)s: %(message)s')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    #simulate_random_poliy(task_start_states, task_state_action_map)

    # normalizing feature expection to bind the first norm of rewards and w within 1
    mu_e = mu_e/np.linalg.norm(mu_e)
    # first iteration
    pi = init_random_policy(task_state_action_map)
    agent_policy = init_random_policy(task_state_action_map)
    mu_curr = get_mu(pi, agent_policy, n_trials)
    i = 1
    mu_bar_curr = mu_curr
    w = (mu_e - mu_bar_curr)
    t = np.linalg.norm(w)
    reward_table = np.reshape(np.dot(phi_matrix, w), (len(task_states), len(ts.task_actions)))
    ######## Use reward_table on mdp to get new policy  ###########
    pi = init_random_policy(task_state_action_map) # this should come from mdp solution
    mu_curr = get_mu(pi, agent_policy, n_trials)
    mu_bar_prev = mu_bar_curr
    while t > 0.1:
        print "Iteration: ", i
        print "mu_bar_prev = ", mu_bar_prev
        x = mu_curr - mu_bar_prev
        y = mu_e - mu_bar_prev
        mu_bar_curr = mu_bar_prev + (np.dot(x.T, y)/np.dot(x.T, x)) * x
        rstate_idx = random.randrange(0, len(task_states))
        print "mu_bar_curr = ", mu_bar_curr
        print "reward[", rstate_idx, "] = ", reward_table[rstate_idx]
        print "||w||_1 = ", np.linalg.norm(w, 1)
        print "t = ", np.round(t, 3)
        w = (mu_e - mu_bar_curr)
        t = np.linalg.norm(w)
        reward_table = np.reshape(np.dot(phi_matrix, w), (len(task_states), len(ts.task_actions)))
        i = i + 1
        ##### Use reward_table on mdp to get new policy  ###########
        pi = init_random_policy(task_state_action_map) # this should come from mdp solution
        mu_curr = get_mu(pi, agent_policy, n_trials)
        mu_bar_prev = mu_bar_curr
        print "**********************************************************"
        user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        if user_input.upper() == 'Q':
            break;
        print "**********************************************************"

if __name__=='__main__':
    main()

