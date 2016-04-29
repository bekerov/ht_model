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
    with open("dists.pickle", "r") as dists_file:
        r1_dists = pickle.load(dists_file)
        r2_dists = pickle.load(dists_file)

    t = extract_best_policy_dict_from_numpy(r1_dists[0])
    q = convert_to_dict_from_numpy(r1_dists[1])
    #pprint.pprint(t)
    #pprint.pprint(q)
    s = ts.State(n_r=3, n_h=0, t_r=0, t_h=1, b_r=1, b_h=0, e=0)
    #print t[s], q[s]
    tidx = task_states_list.index(s)
    #print task_states_list[tidx], t[s], q[s], select_random_action(r1_dists[0][tidx])
    #print r1_dists[0][tidx]
    n = convert_to_numpy_from_dict(q)
    print np.array_equal(r1_dists[1], n)
