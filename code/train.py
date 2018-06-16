import numpy as np

from utils import *
from curriculum import random, update_backward_reachable_set, sample_from_backward_reachable_set, backward_reachable
from problem import Problem
from rl import ppo
from collections import defaultdict
from data_logger import DataLogger
from time import strftime
from random_utils import fixed_random_seed
from plotting_with_theta_sections import visualize_starts, visualize_rollouts, plot_performance

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("type", help="what kind of curriculum to employ",
                    type=str)
parser.add_argument("--debug", help="whether to print loads of information and create plots or not",
                    action="store_true")
parser.add_argument("--brs_sample", help="how to sample from a backreachable set",
                    type=str, default='contour_edges')
parser.add_argument("--run_name", help="name of run that determines where outputs are saved",
                    type=str, default=None)
parser.add_argument("--zero_goal_v", help="whether to make the goal have zero initial velocity",
                    action="store_true")
parser.add_argument("--finish_threshold", help="what fraction of starts must be successful to finish training.",
                    type=float, default=0.95)
parser.add_argument("--seed", help="what seed to use for the random number generators.",
                    type=int, default=2018)
parser.add_argument("--disturbance", help="what disturbance to use in the gym environment.",
                    type=str, default=None)
parser.add_argument("--gym_env", help="which gym environment to use.",
                    type=str, default='DrivingOrigin-v0')
parser.add_argument("--hover_at_end", help="whether to null velocity and rates at the end",
                    action="store_true")
parser.add_argument("--variation", help="what variation to use",
                    type=int, default=None)

args = parser.parse_args()

if args.type in ['backreach', 'random', 'ppo_only']:
    if args.run_name is not None:
        run_dir = args.run_name
    else:
        run_dir = args.type
else:
    parser.error('"%s" is not in ["backreach", "random", "ppo_only"]!' % args.type);

RUN_DIR = os.path.join(os.getcwd(), 'runs', args.gym_env + '_' + run_dir + '_' + strftime('%d-%b-%Y_%H-%M-%S'))
FIGURES_DIR = os.path.join(RUN_DIR, 'figures')
DATA_DIR = os.path.join(RUN_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')

if args.gym_env == 'DrivingOrigin-v0':
    X_IDX = 0
    Y_IDX = 1
    from backreach.car5d_interface import Car5DBackreachEngine as BackreachEngine
elif args.gym_env == 'PlanarQuad-v0':
    X_IDX = 0
    Y_IDX = 2
    from backreach.quad6d_interface import Quad6DBackreachEngine as BackreachEngine


def train_step(weighted_start_states, policy, train_algo, problem,
               num_ppo_iters=20):

    problem.start_state_dist = weighted_start_states
    if problem.env_name == 'PlanarQuad-v0':
        gam = 0.99
    else:
        gam = 0.998

    return train_algo.train(problem, policy, min_cost=problem.min_cost, 
                            timesteps_per_actorbatch=2048,
                            clip_param=0.2, entcoeff=0.0,
                            optim_epochs=10, optim_stepsize=3e-4, 
                            optim_batchsize=64, gamma=gam, lam=0.95,
                            max_iters=num_ppo_iters)


def train(problem,              # Problem object, describing the task.
          initial_policy,       # Initial policy.
          goal_state,           # Goal state (s^g).
          full_start_dist,      # Full start state distribution (rho_0).
          N_new=200, N_old=100, # Num new and old start states to sample.
          R_min=0.5, R_max=1.0, # Reward bounds defining what a "good" start state is.
          num_iters=100,        # Number of iterations of training to run.
          num_ppo_iters=20,     # Number of iterations of PPO training to run per train step.
          curriculum_strategy=random, 
          train_algo=ppo, 
          start_distribution=uniform,
          debug=False):

    data_logger = DataLogger(col_names=['overall_iter', 'ppo_iter', 
                                        'overall_perf', 'overall_area', 
                                        'ppo_perf', 'ppo_lens', 'ppo_rews'],
                             filepath=os.path.join(DATA_DIR, 'data_%s.csv' % args.type),
                             data_dir=DATA_DIR)

    print(locals(), flush=True);
    if curriculum_strategy in [random]:
        return train_pointwise(**locals());
    elif curriculum_strategy in [backward_reachable]:
        return train_gridwise(**locals());
    elif curriculum_strategy is None:
        return train_ppo_only(**locals());
    else:
        raise ValueError("You passed in an unknown curriculum strategy!");


# "PPO only" because we're not doing any curriculum at all.
def train_ppo_only(**kwargs):
    """Train a policy with no curriculum strategy, only the training algo.
    """
    problem = kwargs["problem"];
    initial_policy = kwargs["initial_policy"];
    goal_state = kwargs["goal_state"];
    full_start_dist = kwargs["full_start_dist"];
    N_new, N_old = kwargs["N_new"], kwargs["N_old"];
    R_min, R_max = kwargs["R_min"], kwargs["R_max"];
    num_iters = kwargs["num_iters"];
    num_ppo_iters = kwargs["num_ppo_iters"];
    curriculum_strategy = kwargs["curriculum_strategy"];
    train_algo = kwargs["train_algo"];
    start_distribution = kwargs["start_distribution"];
    debug = kwargs["debug"];
    data_logger = kwargs["data_logger"];

    pi_i = initial_policy
    overall_perf = list()
    ppo_lens, ppo_rews = list(), list()
    perf_metric = 0.0
    i = 0
    ppo_iter_count = 0;
    pi_i.save_model(MODEL_DIR, iteration=i);
    while i < num_iters*num_ppo_iters:
    # while perf_metric < args.finish_threshold and i < num_iters*num_ppo_iters:
        print('Training Iteration %d' % i, flush=True)
        data_logger.update_indices({"overall_iter": i, "ppo_iter": ppo_iter_count})
        pi_i, rewards_map, ep_mean_lens, ep_mean_rews = train_step(full_start_dist, pi_i, train_algo, problem, num_ppo_iters=num_ppo_iters)

        ppo_lens.extend(ep_mean_lens)
        ppo_rews.extend(ep_mean_rews)

        perf_metric = evaluate(pi_i, 
                             full_start_dist, 
                             problem, 
                             debug=debug, 
                             figfile=os.path.join(FIGURES_DIR, 'eval_iter_%d' % i))

        overall_perf.append(perf_metric)

        data_logger.add_rows({'overall_perf': [perf_metric], 'ppo_perf': [perf_metric], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews},
                              update_indices=['overall_iter', 'ppo_iter'])

        if debug:
            plot_performance(range(len(overall_perf)), overall_perf, ylabel=r'% Successful Starts', xlabel='Iteration', figfile=os.path.join(FIGURES_DIR, 'overall_perf'))

        print('[Overall Iter %d]: perf_metric = %.2f' % (i, perf_metric));

        # Incrementing our algorithm's loop counter.
        # Here, these are the same since the algorithm is PPO itself.
        ppo_iter_count += num_ppo_iters;
        i += num_ppo_iters;

        data_logger.save_to_file();
        pi_i.save_model(MODEL_DIR, iteration=i);

    return pi_i


# "Pointwise" because the unit of reachibility here is explicit states,
# i.e. "points in the state space." Our random baseline is point-based.
def train_pointwise(**kwargs):
    """Train a policy with a specific curriculum 
    strategy (random, etc), 
    policy training method (trpo, ppo, etc), and start state sampling 
    strategy (uniform, weighted by value function, etc).
    """
    problem = kwargs["problem"];
    initial_policy = kwargs["initial_policy"];
    goal_state = kwargs["goal_state"];
    full_start_dist = kwargs["full_start_dist"];
    N_new, N_old = kwargs["N_new"], kwargs["N_old"];
    R_min, R_max = kwargs["R_min"], kwargs["R_max"];
    num_iters = kwargs["num_iters"];
    num_ppo_iters = kwargs["num_ppo_iters"];
    curriculum_strategy = kwargs["curriculum_strategy"];
    train_algo = kwargs["train_algo"];
    start_distribution = kwargs["start_distribution"];
    debug = kwargs["debug"];
    data_logger = kwargs["data_logger"];

    # Keyword arguments for the curriculum strategy.
    curric_kwargs = defaultdict(lambda: None)

    # Not used in algorithm, only for visualization.
    all_starts = [goal_state]
    
    old_starts = [goal_state]
    starts = [goal_state]
    pi_i = initial_policy
    overall_perf, overall_area = list(), list()
    ppo_perf, ppo_lens, ppo_rews = list(), list(), list()
    perf_metric = 0.0
    i = 0
    ppo_iter_count = 0;
    pi_i.save_model(MODEL_DIR, iteration=i);
    while i < num_iters:
    # while perf_metric < args.finish_threshold and i < num_iters:
        print('Training Iteration %d' % i, flush=True)
        data_logger.update_indices({"overall_iter": i, "ppo_iter": ppo_iter_count})

        new_starts = curriculum_strategy(starts, N_new, problem)

        if debug and len(new_starts) > 0:
            visualize_starts(new_starts, problem, 
                             figfile=os.path.join(FIGURES_DIR, 'curric_starts_iter_%d' % i))

        from_replay = sample(old_starts, size=N_old)
        starts = new_starts + from_replay

        if debug:
            visualize_starts(new_starts, problem, 
                             old_starts=from_replay,
                             old_start_color='orange',
                             old_start_name='Replay Starts',
                             figfile=os.path.join(FIGURES_DIR, 'replay_and_curric_starts_iter_%d' % i))
            
            visualize_starts(None, problem, 
                             old_starts=all_starts,
                             old_start_color='grey',
                             old_start_name='Old Starts',
                             figfile=os.path.join(FIGURES_DIR, 'previous_starts_iter_%d' % i))

        rho_i = list(zip(starts, start_distribution(starts)))
        pi_i, rewards_map, ep_mean_lens, ep_mean_rews = train_step(rho_i, pi_i, train_algo, problem, num_ppo_iters=num_ppo_iters)

        if debug:
            if problem.env_name == 'DrivingOrigin-v0':
                visualize_starts(starts, problem, 
                                 rewards_map=rewards_map,
                                 figfile=os.path.join(FIGURES_DIR, 'start_rews_iter_%d' % i))

            visualize_rollouts([sample(starts)], pi_i, problem,
                               figfile=os.path.join(FIGURES_DIR, 'rollouts_iter_%d' % i))

        data_logger.save_to_npy('curr_starts', starts);

        all_starts.extend(starts)
        total_unique_starts = len(dedupe_list_of_np_arrays(starts))
        starts = select(starts, rewards_map, R_min, R_max, problem)
        successful_starts = len(starts)
        pct_successful = float(successful_starts)/total_unique_starts;
        old_starts.extend(starts)

        ppo_perf.append(pct_successful*100.)
        ppo_lens.extend(ep_mean_lens)
        ppo_rews.extend(ep_mean_rews)

        if debug:
            plot_performance(range(len(ppo_perf)), ppo_perf, ylabel=r'% Successful Starts', xlabel=r'PPO Iteration ($\times %d$)' % num_ppo_iters, figfile=os.path.join(FIGURES_DIR, 'ppo_pct_succ_iter_%d' % i))
            plot_performance(range(len(ppo_lens)), ppo_lens, ylabel=r'Avg. Episode Length', xlabel='PPO Iteration', figfile=os.path.join(FIGURES_DIR, 'ppo_avg_lens_iter_%d' % i))
            plot_performance(range(len(ppo_rews)), ppo_rews, ylabel=r'Avg. Episode Reward', xlabel='PPO Iteration', figfile=os.path.join(FIGURES_DIR, 'ppo_avg_rews_iter_%d' % i))

        data_logger.save_to_npy('all_starts', all_starts);
        data_logger.save_to_npy('old_starts', old_starts);
        data_logger.save_to_npy('selected_starts', starts);
        data_logger.save_to_npy('new_starts', new_starts);
        data_logger.save_to_npy('from_replay', from_replay);

        ppo_iter_count += num_ppo_iters;

        perf_metric = evaluate(pi_i, 
                             full_start_dist, 
                             problem, 
                             debug=debug, 
                             figfile=os.path.join(FIGURES_DIR, 'eval_iter_%d' % i))

        # Format is (min_x, max_x, min_y, max_y)
        all_starts_bbox = bounding_box(all_starts)
        min_x = problem.state_space.low[X_IDX]
        max_x = problem.state_space.high[X_IDX]
        min_y = problem.state_space.low[Y_IDX]
        max_y = problem.state_space.high[Y_IDX]
        area_coverage = bounding_box_area(all_starts_bbox) / bounding_box_area((min_x, max_x, min_y, max_y))

        overall_perf.append(perf_metric)
        overall_area.append(area_coverage*100.)

        data_logger.add_rows({'overall_perf': [perf_metric], 'overall_area': [area_coverage], 
                              'ppo_perf': [pct_successful], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews},
                              update_indices=['ppo_iter'])

        if debug:
            plot_performance(range(len(overall_perf)), overall_perf, ylabel=r'% Successful Starts', xlabel='Iteration', figfile=os.path.join(FIGURES_DIR, 'overall_perf'))
            plot_performance(range(len(overall_area)), overall_area, ylabel=r'% State Space Sampled', xlabel='Iteration', figfile=os.path.join(FIGURES_DIR, 'overall_area'))

        print('[Overall Iter %d]: perf_metric = %.2f | Area Coverage = %.2f%%' % (i, perf_metric, area_coverage*100.));

        # Incrementing our algorithm's loop counter.
        i += 1;

        data_logger.save_to_file();
        pi_i.save_model(MODEL_DIR, iteration=i);

    return pi_i


# "Gridwise" because the unit of reachibility here is grids of the state space.
# Our backward reachibility method is grid-based.
def train_gridwise(**kwargs):
    """Train a policy with a specific curriculum 
    strategy (backward reachable, etc), 
    policy training method (trpo, ppo, etc), and start state sampling 
    strategy (uniform, weighted by value function, etc).
    """
    problem = kwargs["problem"];
    initial_policy = kwargs["initial_policy"];
    goal_state = kwargs["goal_state"];
    full_start_dist = kwargs["full_start_dist"];
    N_new, N_old = kwargs["N_new"], kwargs["N_old"];
    R_min, R_max = kwargs["R_min"], kwargs["R_max"];
    num_iters = kwargs["num_iters"];
    num_ppo_iters = kwargs["num_ppo_iters"];
    curriculum_strategy = kwargs["curriculum_strategy"];
    train_algo = kwargs["train_algo"];
    start_distribution = kwargs["start_distribution"];
    debug = kwargs["debug"];
    data_logger = kwargs["data_logger"];

    # Keyword arguments for the curriculum strategy.
    curric_kwargs = defaultdict(lambda: None)
    curric_kwargs["debug"] = debug
    if curriculum_strategy == backward_reachable:
        br_engine = BackreachEngine()
        br_engine.reset_variables(problem, os.path.join(FIGURES_DIR, ''))
        curric_kwargs['br_engine'] = br_engine
        curric_kwargs['curr_train_iter'] = 0
        curric_kwargs['brs_sample'] = args.brs_sample
        curric_kwargs['variation'] = args.variation
        curric_kwargs['problem'] = problem

    # Not used in algorithm, only for visualization.
    all_starts = [goal_state]
    
    old_starts = [goal_state]
    starts = [goal_state]
    pi_i = initial_policy
    overall_perf, overall_area = list(), list()
    perf_metric = 0.0
    i = 0
    pi_i.save_model(MODEL_DIR, iteration=i);
    while i < num_iters:
    # while perf_metric < args.finish_threshold and i < num_iters:
        print('Training Iteration %d' % i, flush=True);
        data_logger.update_indices({"overall_iter": i})

        if 'curr_train_iter' in curric_kwargs:
            curric_kwargs['curr_train_iter'] = i;

        # I've split apart the following call into two separate ones.
        # new_starts = curriculum_strategy(starts, N_new, problem, **curric_kwargs)
        if curriculum_strategy == backward_reachable:
            update_backward_reachable_set(starts, **curric_kwargs);
            data_logger.save_to_npy('brs_targets', starts);

        pct_successful = 0.0;
        iter_count = 0;
        ppo_perf, ppo_lens, ppo_rews = list(), list(), list()
        # Think of this as "while (haven't passed this grade)"
        while pct_successful < 0.5:
            data_logger.update_indices({"ppo_iter": iter_count})

            if curriculum_strategy == backward_reachable:
                new_starts = sample_from_backward_reachable_set(N_new, **curric_kwargs);

                if debug:
                    br_engine.visualize_grids(os.path.join(FIGURES_DIR, ''), '_iter_%d_ppo_iter_%d' % (i, iter_count))

            if debug:
                visualize_starts(new_starts, problem, 
                                 figfile=os.path.join(FIGURES_DIR, 'curric_starts_iter_%d_ppo_iter_%d' % (i, iter_count)))

            if args.variation == 2 and br_engine.check_membership(np.array([problem.env.unwrapped.start_state])):
                print('Variation 2 condition train.py!', flush=True)
                from_replay = list()
                starts = [problem.env.unwrapped.start_state]
            else:
                from_replay = sample(old_starts, size=N_old)
                starts = new_starts + from_replay
            
            if debug:
                visualize_starts(new_starts, problem, 
                                 old_starts=from_replay,
                                 old_start_color='orange',
                                 old_start_name='Replay Starts',
                                 figfile=os.path.join(FIGURES_DIR, 'replay_and_curric_starts_iter_%d_ppo_iter_%d' % (i, iter_count)))
                
                visualize_starts(None, problem, 
                                 old_starts=all_starts,
                                 old_start_color='grey',
                                 old_start_name='Old Starts',
                                 old_start_alpha=0.2,
                                 with_arrows=False,
                                 figfile=os.path.join(FIGURES_DIR, 'previous_starts_iter_%d_ppo_iter_%d' % (i, iter_count)))

            rho_i = list(zip(starts, start_distribution(starts)))
            pi_i, rewards_map, ep_mean_lens, ep_mean_rews = train_step(rho_i, pi_i, train_algo, problem, num_ppo_iters=num_ppo_iters)

            if debug:
                if problem.env_name == 'DrivingOrigin-v0':
                    visualize_starts(starts, problem, 
                                     rewards_map=rewards_map,
                                     figfile=os.path.join(FIGURES_DIR, 'start_rews_iter_%d_ppo_iter_%d' % (i, iter_count)))

                visualize_rollouts([sample(starts)], pi_i, problem,
                                  figfile=os.path.join(FIGURES_DIR, 'rollouts_iter_%d_ppo_iter_%d' % (i, iter_count)))

            data_logger.save_to_npy('curr_starts', starts);

            all_starts.extend(starts)
            total_unique_starts = len(dedupe_list_of_np_arrays(starts))
            starts = select(starts, rewards_map, R_min, R_max, problem)
            successful_starts = len(starts)
            pct_successful = float(successful_starts)/total_unique_starts;

            ppo_perf.append(pct_successful*100.)
            ppo_lens.extend(ep_mean_lens)
            ppo_rews.extend(ep_mean_rews)

            data_logger.add_rows({'ppo_perf': [pct_successful], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews}, update_indices=['ppo_iter'])

            if debug:
                plot_performance(range(len(ppo_perf)), ppo_perf, ylabel=r'% Successful Starts', xlabel=r'PPO Iteration ($\times %d$)' % num_ppo_iters, figfile=os.path.join(FIGURES_DIR, 'ppo_pct_succ_iter_%d' % i))
                plot_performance(range(len(ppo_lens)), ppo_lens, ylabel=r'Avg. Episode Length', xlabel='PPO Iteration', figfile=os.path.join(FIGURES_DIR, 'ppo_avg_lens_iter_%d' % i))
                plot_performance(range(len(ppo_rews)), ppo_rews, ylabel=r'Avg. Episode Reward', xlabel='PPO Iteration', figfile=os.path.join(FIGURES_DIR, 'ppo_avg_rews_iter_%d' % i))

            data_logger.save_to_npy('all_starts', all_starts);
            data_logger.save_to_npy('old_starts', old_starts);
            data_logger.save_to_npy('selected_starts', starts);
            data_logger.save_to_npy('new_starts', new_starts);
            data_logger.save_to_npy('from_replay', from_replay);

            iter_count += num_ppo_iters;
            print('[PPO Iter %d]: %.2f%% Successful Starts (%d / %d)' % (iter_count, pct_successful*100., successful_starts, total_unique_starts));

        # This final update is so we get the last iter_count correctly after jumping out of the while loop.
        data_logger.update_indices({"ppo_iter": iter_count})

        # Ok, we've graduated!
        old_starts.extend(starts)
        perf_metric = evaluate(pi_i, 
                             full_start_dist, 
                             problem, 
                             debug=debug, 
                             figfile=os.path.join(FIGURES_DIR, 'eval_iter_%d' % i))

        # Format is (min_x, max_x, min_y, max_y)
        all_starts_bbox = bounding_box(all_starts)
        min_x = problem.state_space.low[X_IDX]
        max_x = problem.state_space.high[X_IDX]
        min_y = problem.state_space.low[Y_IDX]
        max_y = problem.state_space.high[Y_IDX]
        area_coverage = bounding_box_area(all_starts_bbox) / bounding_box_area((min_x, max_x, min_y, max_y))
        
        overall_perf.append(perf_metric)
        overall_area.append(area_coverage*100.)

        data_logger.add_rows({'overall_perf': [perf_metric], 'overall_area': [area_coverage]})

        if debug:
            plot_performance(range(len(overall_perf)), overall_perf, ylabel=r'% Successful Starts', xlabel='Iteration', figfile=os.path.join(FIGURES_DIR, 'overall_perf'))
            plot_performance(range(len(overall_area)), overall_area, ylabel=r'% State Space Sampled', xlabel='Iteration', figfile=os.path.join(FIGURES_DIR, 'overall_area'))

        print('[Overall Iter %d]: perf_metric = %.2f | Area Coverage = %.2f%%' % (i, perf_metric, area_coverage*100.));

        # Incrementing our algorithm's loop counter.
        i += 1;

        data_logger.save_to_file();
        pi_i.save_model(MODEL_DIR, iteration=i);

    # Done!
    if curriculum_strategy == backward_reachable:
        br_engine.stop();
        del br_engine;

    return pi_i


if __name__ == '__main__':
    # Using context managers for random seed management.
    with fixed_random_seed(args.seed):
        maybe_mkdir(RUN_DIR);
        maybe_mkdir(DATA_DIR);
        maybe_mkdir(FIGURES_DIR);
        maybe_mkdir(MODEL_DIR);

        # Keeping dist same across runs for comparison.
        with fixed_random_seed(2018):
            problem = Problem(args.gym_env, 
                              zero_goal_v=args.zero_goal_v, 
                              disturbance=args.disturbance)

            if args.gym_env == 'DrivingOrigin-v0':
                num_iters = 100
                print('Goal State:', problem.goal_state, flush=True)
                full_starts = problem.sample_behind_goal(problem.goal_state,
                                                         num_states=100, 
                                                         zero_idxs=[3, 4])
            elif args.gym_env == 'PlanarQuad-v0':
                num_iters = 40
                full_starts = [problem.env.unwrapped.start_state]
                problem.env.unwrapped.set_hovering_goal(args.hover_at_end)

        full_start_dist = list(zip(full_starts, uniform(full_starts)))

        ppo.create_session(num_cpu=1)
        initial_policy = ppo.create_policy('pi', problem)
        ppo.initialize()

        if args.type == 'random':
            curr_strategy = random
        elif args.type == 'backreach':
            curr_strategy = backward_reachable
        elif args.type == 'ppo_only':
            curr_strategy = None
        else:
            raise ValueError("%s is an unknown curriculum strategy!" % args.type);

        trained_policy = train(problem=problem,
                               num_iters=num_iters,
                               R_min=problem.R_min, R_max=problem.R_max,
                               initial_policy=initial_policy,
                               goal_state=problem.goal_state,
                               full_start_dist=full_start_dist,
                               curriculum_strategy=curr_strategy,
                               debug=args.debug)

        trained_policy.save_model(MODEL_DIR);
        print('Done training!');
