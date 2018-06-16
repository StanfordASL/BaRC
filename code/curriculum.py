from utils import uniform, sample
from numpy.random import uniform
import numpy as np


def random(starts, N_new, problem, **kwargs):
    if len(starts) == 0:
        print('empty starts list!', flush=True)
        return list()
    
    num_random_steps = 10 if 'num_random_steps' not in kwargs else kwargs['num_random_steps']
    total_runs = 1000 if 'total_runs' not in kwargs else kwargs['total_runs']
    
    while len(starts) < total_runs:
        start = sample(starts)
        problem.reset_to_state(start)
        for t in range(num_random_steps):
            action_t = uniform(low=problem.action_space.low,
                               high=problem.action_space.high)

            state_t, _, _, _ = problem.step(action_t, ret_state=True)

            if (problem.env_name == 'PlanarQuad-v0' 
                and problem.env.unwrapped._in_obst(state_t)):
                    break

            starts.append(state_t)

    new_starts = sample(starts, size=N_new)
    return new_starts


def update_backward_reachable_set(starts, **kwargs):
    br_engine = kwargs['br_engine'];
    problem = kwargs['problem']
    variation = kwargs['variation']

    # (1) Check if start is in backreach set (if so, stop expanding).
    if variation == 1 and br_engine.check_membership(np.array([problem.env.unwrapped.start_state])):
        print('Variation 1 condition curriculum.py!', flush=True)
        return

    br_engine.update_and_compute_backward_reachable_set(starts, 
                                                        plot=kwargs['debug'],
                                                        curr_train_iter=kwargs['curr_train_iter']);


def sample_from_backward_reachable_set(N_new, **kwargs):
    br_engine = kwargs['br_engine'];
    problem = kwargs['problem']
    variation = kwargs['variation']

    # (2) When you do reach it, only sample the start
    if variation == 2 and br_engine.check_membership(np.array([problem.env.unwrapped.start_state])):
        print('Variation 2 condition curriculum.py!', flush=True)
        return [problem.env.unwrapped.start_state]

    new_starts_arr = br_engine.sample_from_grid(size=N_new, method=kwargs['brs_sample']);
    
    return [new_starts_arr[i] for i in range(new_starts_arr.shape[0])];


def backward_reachable(starts, N_new, problem, **kwargs):
    update_backward_reachable_set(starts, **locals());
    return sample_from_backward_reachable_set(N_new, **locals());
