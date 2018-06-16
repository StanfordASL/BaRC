import pandas as pd
import numpy as np
import plotting_with_arrows as plotting_lib
import plotting_performance as perf_plot
# import plotting as plotting_lib


def plot_performance(x, y, **kwargs):
    perf_plot.plot_performance(x, y, **kwargs);


def visualize_rollouts(starts, policy, problem, 
                       **kwargs):
    plotting_lib.visualize_rollouts(starts, policy, problem,
                                    **kwargs);


def visualize_starts(starts, problem, bins=0, **kwargs):
    # Currently plotting both versions.
    plotting_lib.visualize_starts(starts, problem, **kwargs);

    if bins > 0: # This will be a problem if bins is ever > 0, thankfully by default it isn't.
        starts_df = pd.DataFrame(np.stack(starts), index=range(len(starts)));
        starts_df.columns = ['x', 'y', 'theta', 'v', 'w'];

        starts_df['theta_bin'] = pd.cut(starts_df['theta'], bins=bins);

        for idx, bin in enumerate(sorted(pd.unique(starts_df['theta_bin']))):
            curr_kwargs = kwargs.copy()
            bin_name = str(bin)
            curr_kwargs['title_adj'] = '\n' + r'$\theta \in$ ' + bin_name
            curr_kwargs['figfile'] += '_theta_%d' % idx;
            starts_with_theta = starts_df[starts_df['theta_bin'] == bin]
            del starts_with_theta['theta_bin']
            plotting_lib.visualize_starts(starts_with_theta.values.tolist(),
                                          problem,
                                          **curr_kwargs);


def visualize_eval(starts, problem, results, bins=0, **kwargs):
    # Currently plotting both versions.
    plotting_lib.visualize_eval(starts, problem, results, **kwargs);

    if bins > 0: # This will be a problem if bins is ever > 0, thankfully by default it isn't.
        starts_df = pd.DataFrame(np.stack(starts), index=range(len(starts)));
        starts_df.columns = ['x', 'y', 'theta', 'v', 'w'];

        starts_df['theta_bin'] = pd.cut(starts_df['theta'], bins=bins);

        for idx, bin in enumerate(sorted(pd.unique(starts_df['theta_bin']))):
            curr_kwargs = kwargs.copy()
            bin_name = str(bin)
            curr_kwargs['title_adj'] = '\n' + r'$\theta \in$ ' + bin_name
            curr_kwargs['figfile'] += '_theta_%d' % idx;
            idx_mask = starts_df['theta_bin'] == bin
            plotting_lib.visualize_eval(starts_df[idx_mask].values.tolist(),
                                        problem, 
                                        np.asarray(results)[np.where(idx_mask)[0]],
                                        **curr_kwargs);


if __name__ == '__main__':
    from problem import Problem

    problem = Problem('DrivingOrigin-v0');
    starts = problem.sample_from_space(num_states=100);
    visualize_starts(starts, problem, bins=10, figfile='figures/test');
    visualize_eval(starts, problem, [0]*50 + [1]*50, bins=10, figfile='figures/eval');
