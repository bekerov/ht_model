#!/usr/bin/env python

import sys
import logging
import cPickle as pickle
import random
import numpy as np

from termcolor import colored

import taskSetup as ts
import simulationFunctions as sf

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

    # we are ignoring the exit task bit (doesn't add any information, always 1 on mu_e) so nstates-1
    mu = np.zeros(ts.n_states-1)
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
            mu = mu + ts.get_phi(list(state_r1))
    return mu/n_trials

def get_phi(task_state_vector, current_action):
    """ Function to return the feature vector given the state vector and action.
    Arg:
        List: state vector
    Return
        np_array: phi
    """
    state_feature = [1 if task_state_vector[0] else 0] + [1 if task_state_vector[1] else 0] + task_state_vector[2:]
    action_feature = [1 if action == current_action else 0 for action in list(ts.task_actions)]
    feature_vector = state_feature + action_feature
    return np.array(feature_vector)

if __name__=='__main__':
    logging.basicConfig(level=logging.WARN, format='%(asctime)s-%(levelname)s: %(message)s')
    #task_states, task_start_states, task_state_action_map, phi_state = ts.load_state_data()
    task_states, task_start_states, task_state_action_map = ts.load_state_data()
    task_actions = list(ts.task_actions)
    task_states = list(task_states)
    t = random.choice(task_states)
    a = random.choice(task_actions)
    phi = np.empty((0, (ts.n_states+len(task_actions))))
    for task_state in task_states:
        for task_action in task_actions:
            phi = np.vstack((phi, get_phi(list(task_state), task_action)))
    #np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
    print phi
    print phi.shape, phi.size
    phi = np.reshape(phi, (len(task_states), len(task_actions), (ts.n_states + len(task_actions))))
    print phi
    print phi.shape, phi.size

    #_, _, mu_e, n_trials = ts.read_task_data()
    #simulate_random_poliy(task_start_states, task_state_action_map)

    # first iteration
    #pi = init_random_policy(task_state_action_map)
    #agent_policy = init_random_policy(task_state_action_map)
    #mu_curr = get_mu(pi, agent_policy, n_trials)
    #i = 1
    #mu_bar_curr = mu_curr
    #w = mu_e - mu_bar_curr
    #t = np.linalg.norm(w)
    #rewards = np.dot(w, phi_state)
    ###### Use rewards on mdp to get new policy  ###########
    #pi = init_random_policy(task_state_action_map) # this should come from mdp solution
    #mu_curr = get_mu(pi, agent_policy, n_trials)
    #mu_bar_prev = mu_bar_curr
    #while True:
        #print "Iteration: ", i
        #print "mu_bar_prev = ", mu_bar_prev
        #x = mu_curr - mu_bar_prev
        #y = mu_e - mu_bar_prev
        #mu_bar_curr = mu_bar_prev + (np.dot(x.T, y)/np.dot(x.T, x)) * x
        #print "mu_bar_curr = ", mu_bar_curr
        #print "t = ", t
        #w = mu_e - mu_bar_curr
        #t = np.linalg.norm(w)
        #rewards = np.dot(w, phi_state)
        #i = i + 1
        ###### Use rewards on mdp to get new policy  ###########
        #pi = init_random_policy(task_state_action_map) # this should come from mdp solution
        #mu_curr = get_mu(pi, agent_policy, n_trials)
        #mu_bar_prev = mu_bar_curr
        #print "**********************************************************"
        #user_input = raw_input('Press Enter to continue, Q-Enter to quit\n')
        #if user_input.upper() == 'Q':
            #break;
        #print "**********************************************************"

    #w_1 = np.absolute(mu_e - mu_0)
    #rewards = np.dot(w_1, phi_state)

    ##### Use rewards on mdp to get new policy  ###########
    #pi_1 = init_random_policy(task_state_action_map) # this should come from mdp solution
    #mu_i = get_mu(pi_1, agent_policy, n_trials)
    #i = 1
    #np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    #print "mu_e = ", mu_e
    #print "mu = ", mu
    #print "w = ", w
    #print "t = ", np.round(t, 3)

