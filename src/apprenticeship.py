#!/usr/bin/env python

# Imports here

EPSILON = 0.5

state-action_space = {}

def init_state-action_space()
    """
    Function:
        Load state-action space from file
    """


def init_policy():
    """
    Function:
        Randomly initialize possible action probabilities for each state.
    Param: 
        s-a_space
            Dictionary mapping each state to possbile actions taken at that
            state.
    Return:
        policy
            Dictionary of dictionaries, mapping each state to possible actions,
            and each action to the probability of choosing that action at that 
            state.
    """
    policy = state-action_space 
    return policy

def update_t_w( mu )
    """
    Function:
        Updates t and w from function described in paper.
    Param: 
        Array of previous expected feature vectors.
    Return:
        New t
        New w
    """


if __name__ == '__main__':
    # Create state-action space
    init_state-action_space()

    # Randomly pick initial policy
    policy = init_polict()

    # Ensure we enter the loop at least one
    t = EPSILON + 1

    while t > EPSILON : 
    # Update t, w
    t, w = update_t_w( mu )

    #3. If t_i < epsilon : Terminate
    if t < EPSILON
        return 

    #4. Use RL to compute optimal policy pi_i given reward R = w_i * phi

    #5. Compute mu( pi_i ) 

    #6. i = i+1 : Go to 2. 
