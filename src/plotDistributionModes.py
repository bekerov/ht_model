#!/usr/bin/env python

import sys
import os
import random
import logging
import numpy as np
import cPickle as pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from termcolor import colored
from scipy import stats

import simulationFunctions as sf
from loadTaskParams import *

# set up logging
logging.basicConfig(format='')
lgr = logging.getLogger("plotDistributionModes.py")
lgr.setLevel(level=logging.INFO)

if __name__=='__main__':
    lgr.info("Loading modes.pickle file")
    with open("modes.pickle", "r") as modes_file:
        all_modes = pickle.load(modes_file)
        n_trials = pickle.load(modes_file)

    for i in range(len(task_start_states_list)):
        plt.subplot(2, 2, i+1)
        plt.plot(range(len(all_modes[i])), all_modes[i], 'ro')
        l = all_modes[i].min()
        u = all_modes[i].max()
        d = (u-l)/8
        lb = l - d
        ub = u + d
        plt.axis([0, len(all_modes[i]), lb, ub])
        plt.title('Start State: %s' % str(task_start_states_list[i]))
        plt.xlabel('Distribution Number')
        plt.ylabel('Mode Value')
        plt.grid()

    plt.suptitle('Modes of various state action distributions for each start state for %d trials' % n_trials)
    plt.show()

