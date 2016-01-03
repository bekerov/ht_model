#!/usr/bin/env python

import sys
import random
import logging
import numpy as np

import commonHumanPolicy
import learnedRobotPolicy
import taskSetup as ts
import simulationFunctions as sf

if __name__=='__main__':
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s-%(levelname)s: %(message)s')
    task_states, task_start_states, task_state_action_map = ts.load_states()
    expert_visited_states, expert_state_action_map, time_per_step = commonHumanPolicy.read_data(task_states, task_start_states)
    ntrials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    nactions_expert = np.zeros(ntrials)
    nactions_random = np.zeros(ntrials)
    for i in range(ntrials):
        start_state = random.choice(tuple(task_start_states))
        expert_policy = commonHumanPolicy.get_common_policy(task_state_action_map, expert_state_action_map)
        random_policy = learnedRobotPolicy.init_random_policy(task_state_action_map)
        nactions_expert[i] = sf.run_simulation(expert_policy, expert_policy, start_state)
        nactions_random[i] = sf.run_simulation(random_policy, random_policy, start_state)
    print "Number of trials = ", ntrials
    print "Metric: Number of action per trial"
    print "********************************************************************************"
    print "                Expert Policy               Random Policy"
    print "********************************************************************************"
    print "Min:\t\t ",format(np.amin(nactions_expert), '.3f'), "\t\t\t", format(np.amin(nactions_random), '.3f')
    print "Max:\t\t ",format(np.amax(nactions_expert), '.3f'), "\t\t\t", format(np.amax(nactions_random), '.3f')
    print "Mean:\t\t ",format(np.mean(nactions_expert), '.3f'), "\t\t\t", format(np.mean(nactions_random), '.3f')
    print "Median:\t\t ",format(np.median(nactions_expert), '.3f'), "\t\t\t", format(np.median(nactions_random), '.3f')
    print "Var:\t\t ", format(np.var(nactions_expert), '.3f'), "\t\t\t", format(np.var(nactions_random), '.3f')
    print "Std:\t\t ", format(np.std(nactions_expert), '.3f'), "\t\t\t", format(np.std(nactions_random), '.3f')

#Number of trials =  1000000
#Metric: Number of action per trial
#********************************************************************************
                #Expert Policy               Random Policy
#********************************************************************************
#Min:              10.000                        10.000
#Max:              36.000                        865966.000
#Mean:             16.244                        22.021
#Median:           16.000                        17.000
#Var:              19.969                        1228522.234
#Std:              4.469                         1108.387

