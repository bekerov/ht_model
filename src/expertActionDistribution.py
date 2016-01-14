#!/usr/bin/env python

import random
import logging

import taskSetup as ts
import simulationFunctions as sf

"""This module creates an expert action distribution for the box color sort task
   based on the most frequent actions taken by humans given a particular
   state.
"""

# load task params from pickle file
task_params = ts.load_task_parameters()
task_states_set = task_params[ts.TaskParams.task_states_set]
task_start_state_set = task_params[ts.TaskParams.task_start_state_set]
task_state_action_dict = task_params[ts.TaskParams.task_state_action_dict]
feature_matrix = task_params[ts.TaskParams.feature_matrix]
expert_visited_states_set = task_params[ts.TaskParams.expert_visited_states_set]
expert_state_action_dict = task_params[ts.TaskParams.expert_state_action_dict]
n_episodes = task_params[ts.TaskParams.n_episodes]
time_per_step = task_params[ts.TaskParams.time_per_step]

def compute_expert_state_action_distribution_dict():
    """Function to compute expert Q value based on the processed video files.
       Note: It is likely that the experts don't visit all the possible states in the state space, thus for states not visited by experts we assign an action based on a uniform distribution for that state. For states visited by experts, assign values for actions based on their frequency.
    """
    expert_state_action_distribution_dict = dict()
    # intially setup the policy from a uniform distribution of actions for states
    for task_state_tup, actions_dict in task_state_action_dict.items():
        expert_actions = dict()
        for action in actions_dict:
            expert_actions[action] = 1.0/len(actions_dict)
        expert_state_action_distribution_dict[task_state_tup] = expert_actions

    # overrite the states visited by the experts by the common actions taken
    for task_state_tup, expert_actions in expert_state_action_dict.items():
        actions_dict = {k: float(v)/total for total in (sum(expert_actions.values()),) for k, v in expert_actions.items()}
        expert_state_action_distribution_dict[task_state_tup] = actions_dict

    return expert_state_action_distribution_dict


def simulate_expert_state_action_distribution_dict():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')
    r1_state_action_distribution_dict = compute_expert_state_action_distribution_dict()
    r2_state_action_distribution_dict = compute_expert_state_action_distribution_dict()
    print "Total number of actions by agents using expert policy is %d" % sf.run_simulation(r1_state_action_distribution_dict,r2_state_action_distribution_dict, random.choice(tuple(task_start_state_set)))

if __name__=='__main__':
    simulate_expert_state_action_distribution_dict()
