import os, errno
import numpy as np
from numpy.random import choice
from plotting_with_theta_sections import visualize_eval
from rl import rl_utils

#############
# Selecting #
#############
def dedupe_list_of_np_arrays(lst):
    return list(map(np.asarray, set(map(tuple, lst))))


def select(starts, rewards_map, R_min, R_max, problem):
    ret = list()
    for start in starts:
        start_str = start.tostring()
        if problem.env_name == 'DrivingOrigin-v0':
            if (start_str in rewards_map 
                and R_min <= rewards_map[start_str] <= R_max):
                ret.append(start)
        else:
            if start_str in rewards_map and rewards_map[start_str] > 0:
                ret.append(start)

    return dedupe_list_of_np_arrays(ret)


########################
# Point Bounding Boxes #
########################
def bounding_box(points_list, x_idx=0, y_idx=1):
    points_mat = np.stack(points_list);
    max_values = points_mat.max(axis=0);
    min_values = points_mat.min(axis=0);
    
    # Format is (min_x, max_x, min_y, max_y)
    return (min_values[x_idx], 
            max_values[x_idx], 
            min_values[y_idx], 
            max_values[y_idx])


def bounding_box_area(bounding_box):
    # Format is (min_x, max_x, min_y, max_y)
    return (bounding_box[3] - bounding_box[2])*(bounding_box[1] - bounding_box[0])


############
# Sampling #
############
# Distribution types
def uniform(lst):
    return [1./len(lst)]*len(lst)


# Sampling methods
def sample(lst, distribution=None, size=1):
    if len(lst) == 0:
        raise ValueError('You passed an empty list to be sampled!\nThis usually happens when all states are\noutside of the [0.1, 0.9] reward range.')

    if size in [None, 1]:
        return lst[choice(len(lst), size=None, p=distribution)]

    return [lst[i] for i in choice(len(lst), size=size, p=distribution)]


def weighted_sample(weighted_lst, size=1):
    [starts, distribution] = list(zip(*weighted_lst))
    if size == 1:
        size = None
    return sample(starts, distribution, size=size)


##############
# Evaluation #
##############
def evaluate(policy, weighted_start_states, problem, 
             debug=False, figfile=None):
    if problem.env_name == 'DrivingOrigin-v0':
        start_states = list()
        results = list()
        
        for (start_state, prob) in weighted_start_states:
            reached_goal = rl_utils.rollout(policy, start_state, problem)
            start_states.append(start_state)
            results.append(int(reached_goal))

        if debug:
            visualize_eval(start_states, problem, results,
                           figfile=figfile, make_eps_v=True)

        return np.mean(results)*100.

    elif problem.env_name == 'PlanarQuad-v0':
        _, _, _, rewards = rl_utils.rollout(policy, problem.env.unwrapped.start_state, problem, return_rewards=True)
        return np.sum(rewards)


#################
# OS Operations #
#################
def maybe_mkdir(dirname):
    # Thread-safe way to create a directory if one doesn't exist
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    print('Successfully created', dirname, flush=True)


########
# Math #
########
def signed_delta_angle(x, y):
    return np.arctan2(np.sin(x - y), np.cos(x - y))


if __name__ == '__main__':
    from collections import Counter

    A = ['a', 'b', 'c', 'd']

    rho_i = uniform(A)

    print(rho_i)
    print(Counter(sample(A, rho_i, size=100)).items())
