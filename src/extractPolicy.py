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
lgr = logging.getLogger("extractPolicy.py")
lgr.setLevel(level=logging.INFO)

if __name__ == "__main__":
    lgr.info("Loading agent 1 best distribution dictionary")
    with open("r1_agent_best_dists_dict.pickle", "r") as state_action_dict_file:
        r1_best_dists_dict = pickle.load(state_action_dict_file)
    lgr.info("Loading agent 2 best distribution dictionary")
    with open("r2_agent_best_dists_dict.pickle", "r") as state_action_dict_file:
        r2_best_dists_dict = pickle.load(state_action_dict_file)

    #start_state = random.choice(task_start_states_list)
    for start_state in task_start_states_list:
        lgr.info("start_state = %s", start_state)
        r1_learned_state_action_distribution_dict = random.choice(r1_best_dists_dict[start_state])
        r2_learned_state_action_distribution_dict = random.choice(r2_best_dists_dict[start_state])

        lgr.info("len(r1) = %d, len(r2) = %d", len(r1_learned_state_action_distribution_dict), len(r1_learned_state_action_distribution_dict))

