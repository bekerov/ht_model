#!/usr/bin/env python

actions = {
           'W': 'Wait for human',
           'T': 'Take from table',
           'G': 'Give to human',
           'K': 'Keep on table',
           'R': 'Receive from human',
           'X': 'Exit task'
           }

MAX_OBJECTS = 8
MAX_OBJECTS_ACC = MAX_OBJECTS/2

class TaskState(object):
    """ A class to represent the current state of the task
    """
    n = MAX_OBJECTS_ACC # number of objects on robots side of the table
    r_t = False # Is the robot transferring?
    h_t = False # Is the human transferring?
    r_o = False # Does robot have its object?
    h_o = False # Does robot have the human's object?
    e = False # Has the task ended?

    def set_state(self, states):
        self.n = states[0]
        self.r_t = states[1]
        self.h_t = states[2]
        self.r_o = states[3]
        self.h_o = states[4]
        self.e = states[5]

    def __str__(self):
        state = [str(self.n), str(int(self.r_t)), str(int(self.h_t)), str(int(self.r_o)), str(int(self.h_o)), str(int(self.e))]
        return "state = [" + ', '.join(state) + "]"

class Task(object):
    """ A class to represent the task itself, which has access
    the state of the task
    """
    current_state = TaskState()
    current_action = 'T'
    n_time_steps = 0
    n_time_seconds = 0.0

if __name__=='__main__':
    state = TaskState()
    print state


