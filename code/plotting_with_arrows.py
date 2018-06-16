import matplotlib
matplotlib.use('Agg');

from matplotlib import rc
rc('font',**{'family':'serif'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerRegularPolyCollection
from rl import rl_utils

DPI = 100


class ScatterHandler(HandlerRegularPolyCollection):
    def update_prop(self, legend_handle, orig_handle, legend):
        legend._set_artist_props(legend_handle)
        legend_handle.set_clip_box(None)
        legend_handle.set_clip_path(None)

    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        p = type(orig_handle)([orig_handle.get_paths()[0]],
                              sizes=sizes, offsets=offsets,
                              transOffset=transOffset,
                              cmap=orig_handle.get_cmap(),
                              norm=orig_handle.norm )

        a = orig_handle.get_array()
        if type(a) != type(None) and len(a) > 1:
            p.set_array(np.linspace(0.2, 1.0, len(offsets)))
        else:
            self._update_prop(p, orig_handle)
        return p


def visualize_starts(starts, problem, 
                     old_starts=None,
                     rewards_map=None,
                     edgecolor='k', size=15, edgewidths=0.2,
                     goal_color='b', old_start_color='grey',
                     old_start_alpha=1.0, with_arrows=True,
                     old_start_name='Old Starts', title_adj='',
                     figfile=None):

    print('visualize_starts', flush=True);

    if problem.env_name == 'DrivingOrigin-v0':
        x_idx = 0
        y_idx = 1
        theta_idx = 2
        v_idx = 3
    elif problem.env_name == 'PlanarQuad-v0':
        x_idx = 0
        y_idx = 2
        phi_idx = 4
        vx_idx = 1
        vy_idx = 3


    def plot_arrows(ax, states, color, 
                    head_width=0.05, 
                    head_length=0.05, 
                    alpha=0.5, zorder=1):
        for i in range(states.shape[0]):
            if isinstance(color, str):
                color_val = color
                alpha_val = alpha
            else:
                color_val = 'g'
                alpha_val = alpha*color[i]
	
            if problem.env_name == 'DrivingOrigin-v0':
                ax.arrow(x=states[i, x_idx], y=states[i, y_idx],
                    dx=states[i, v_idx]*np.cos(states[i, theta_idx]),
                    dy=states[i, v_idx]*np.sin(states[i, theta_idx]),
                    head_width=head_width, head_length=head_length, 
                    fc=color_val, ec=color_val, alpha=alpha_val, zorder=zorder);
            elif problem.env_name == 'PlanarQuad-v0':
                ax.arrow(x=states[i, x_idx], y=states[i, y_idx],
                    dx=states[i, vx_idx],
                    dy=states[i, vy_idx],
                    head_width=head_width, head_length=head_length, 
                    fc=color_val, ec=color_val, alpha=alpha_val, zorder=zorder);

    goal_state = problem.goal_state

    fig, ax = plt.subplots()

    # Old Starts
    if old_starts is not None:
        old_xy = np.stack(old_starts)

        if with_arrows:
            plot_arrows(ax, old_xy, old_start_color, alpha=0.5*old_start_alpha);

        ax.plot(old_xy[:, x_idx], old_xy[:, y_idx], alpha=old_start_alpha,
                c=old_start_color, marker='o', markeredgecolor=edgecolor,
                markersize=np.sqrt(size), label=old_start_name, linestyle='None',
                markeredgewidth=edgewidths);

    # Starts
    cm = None
    if starts is not None and len(starts) > 0:
        xy = np.stack(starts)
        if rewards_map is None:
            starts_color = 'g'
            cm = None
        else:
            cm = plt.cm.get_cmap('Greens')
            starts_color = list()
            for start in starts:
                start_str = np.asarray(start).tostring()
                starts_color.append(rewards_map[start_str] 
                                        if start_str in rewards_map else 0.0)

        if with_arrows:
            plot_arrows(ax, xy, starts_color);

        sc = ax.scatter(xy[:, x_idx], xy[:, y_idx],
                        c=starts_color, edgecolor=edgecolor,
                        s=size, label='Starts' if old_starts is None else 'New Starts',
                        cmap=cm, vmin=0, vmax=1,
                        linewidths=edgewidths, alpha=0.8)
        
        if cm is not None:
            fig.colorbar(sc)

    # Goal State
    if problem.env_name == 'DrivingOrigin-v0':
        plot_arrows(ax, np.array([goal_state]), goal_color, zorder=10);
        ax.plot(goal_state[x_idx], goal_state[y_idx],
                    c=goal_color, marker='o', markeredgecolor=edgecolor,
                    markersize=np.sqrt(size), label='Goal', linestyle='None',
                    markeredgewidth=edgewidths, zorder=11);

    elif problem.env_name == 'PlanarQuad-v0':
        for xo,yo,ro in zip(problem.env.unwrapped.obst_X, problem.env.unwrapped.obst_Y, problem.env.unwrapped.obst_R):
            c = plt.Circle((xo,yo),ro, color='black', alpha=1.0)
            ax.add_artist(c)
       
        r = plt.Rectangle((problem.env.unwrapped.xg_lower, problem.env.unwrapped.yg_lower),
                          problem.env.unwrapped.xg_upper - problem.env.unwrapped.xg_lower,
                          problem.env.unwrapped.yg_upper - problem.env.unwrapped.yg_lower,
                          color=goal_color, 
                          alpha=0.3, 
                          hatch='/')
        ax.add_artist(r)

    # ax.set_ylim(problem.observation_space.low[y_idx],  # Bottom
    #             problem.observation_space.high[y_idx]) # Top
    # ax.set_xlim(problem.observation_space.low[x_idx],  # Bottom
    #             problem.observation_space.high[x_idx]) # Top
    ax.axis('equal')
    ax.set_xlabel('$x$ position')
    ax.set_ylabel('$y$ position')

    if problem.env_name == 'DrivingOrigin-v0':
        x, y, th, v, kap = goal_state
        ax.set_title(r'Goal State: [$x$ = %.2f, $y$ = %.2f, $\theta$ = %.2f, $v$ = %.2f, $\kappa$ = %.2f]' % (x, y, th, v, kap) + title_adj)
    elif problem.env_name == 'PlanarQuad-v0':
        ax.set_title(title_adj)

    if starts is not None and cm is not None:
        legend_dict = dict(ncol=1, loc='best', scatterpoints=4, framealpha=1.0)
        ax.legend(handler_map={type(sc) : ScatterHandler()}, **legend_dict)
    else:
        ax.legend(loc='best', framealpha=1.0)

    if figfile is None:
        plt.show()
    else:
        fig.savefig(figfile + '.pdf', dpi=DPI, transparent=True)
        plt.close(fig)

    print('visualize_starts end', flush=True);


def visualize_eval(starts, problem, results,
                   edgecolor='k', size=15, edgewidths=0.2,
                   goal_color='b', title_adj='',
                   figfile=None,
                   make_eps_v=False):
    print('visualize_eval', flush=True);

    if problem.env_name == 'DrivingOrigin-v0':
        x_idx = 0
        y_idx = 1
        theta_idx = 2
        v_idx = 3
    elif problem.env_name == 'PlanarQuad-v0':
        x_idx = 0
        y_idx = 2
        phi_idx = 4
        vx_idx = 1
        vy_idx = 3


    def plot_arrows(ax, states, color, 
                    head_width=0.1, 
                    head_length=0.1, 
                    alpha=0.5,
                    make_eps_v=False):

        if make_eps_v and len(states) > 0:
            if problem.env_name == 'DrivingOrigin-v0':
                states[:, v_idx] = 1e-4

        for i in range(len(states)):
            if isinstance(color, str):
                color_val = color
                alpha_val = alpha
            else:
                color_val = 'g'
                alpha_val = alpha*color[i]

            if problem.env_name == 'DrivingOrigin-v0':
                ax.arrow(x=states[i, x_idx], y=states[i, y_idx],
                    dx=states[i, v_idx]*np.cos(states[i, theta_idx]),
                    dy=states[i, v_idx]*np.sin(states[i, theta_idx]),
                    head_width=head_width, head_length=head_length, 
                    fc=color_val, ec=color_val, alpha=alpha_val);
            elif problem.env_name == 'PlanarQuad-v0':
                ax.arrow(x=states[i, x_idx], y=states[i, y_idx],
                    dx=states[i, vx_idx],
                    dy=states[i, vy_idx],
                    head_width=head_width, head_length=head_length, 
                    fc=color_val, ec=color_val, alpha=alpha_val);


    goal_state = problem.goal_state

    fig, ax = plt.subplots()

    # Starts
    successful_xy = list()
    failed_xy = list()
    for i, start in enumerate(starts):
        if problem.R_min <= results[i] <= problem.R_max:
            successful_xy.append(start)
        else:
            failed_xy.append(start)

    ### Failed Starts
    if len(failed_xy) > 0:
        failed_xy = np.stack(failed_xy)
        plot_x = failed_xy[:, x_idx]
        plot_y = failed_xy[:, y_idx]
    else:
        failed_xy = list()
        plot_x = list()
        plot_y = list()
   
    plot_arrows(ax, failed_xy, 'r', alpha=0.8, make_eps_v=make_eps_v)
    ax.plot(plot_x, plot_y,
            c='r', marker='o', markeredgecolor=edgecolor,
            markersize=np.sqrt(size), label='Failed Starts', linestyle='None',
            markeredgewidth=edgewidths);

    ### Successful Starts
    if len(successful_xy) > 0:
        successful_xy = np.stack(successful_xy)
        plot_x = successful_xy[:, x_idx]
        plot_y = successful_xy[:, y_idx]
    else:
        successful_xy = list()
        plot_x = list()
        plot_y = list()

    plot_arrows(ax, successful_xy, 'g', alpha=0.8, make_eps_v=make_eps_v)
    ax.plot(plot_x, plot_y,
            c='g', marker='o', markeredgecolor=edgecolor,
            markersize=np.sqrt(size), label='Successful Starts', linestyle='None',
            markeredgewidth=edgewidths);

    # Goal State
    if problem.env_name == 'DrivingOrigin-v0':
        plot_arrows(ax, np.array([goal_state]), goal_color);
        ax.plot(goal_state[x_idx], goal_state[y_idx],
                    c=goal_color, marker='o', markeredgecolor=edgecolor,
                    markersize=np.sqrt(size), label='Goal', linestyle='None',
                    markeredgewidth=edgewidths);

    elif problem.env_name == 'PlanarQuad-v0':
        for xo,yo,ro in zip(problem.env.unwrapped.obst_X, problem.env.unwrapped.obst_Y, problem.env.unwrapped.obst_R):
            c = plt.Circle((xo,yo),ro, color='black', alpha=1.0)
            ax.add_artist(c)
       
        r = plt.Rectangle((problem.env.unwrapped.xg_lower, problem.env.unwrapped.yg_lower),
                          problem.env.unwrapped.xg_upper - problem.env.unwrapped.xg_lower,
                          problem.env.unwrapped.yg_upper - problem.env.unwrapped.yg_lower,
                          color=goal_color, 
                          alpha=0.3, 
                          hatch='/')
        ax.add_artist(r)


    # ax.set_ylim(problem.observation_space.low[y_idx],  # Bottom
    #             problem.observation_space.high[y_idx]) # Top
    # ax.set_xlim(problem.observation_space.low[x_idx],  # Bottom
    #             problem.observation_space.high[x_idx]) # Top
    ax.axis('equal')
    ax.set_xlabel('$x$ position')
    ax.set_ylabel('$y$ position')

    if problem.env_name == 'DrivingOrigin-v0':
        x, y, th, v, kap = goal_state
        ax.set_title(r'Goal State: [$x$ = %.2f, $y$ = %.2f, $\theta$ = %.2f, $v$ = %.2f, $\kappa$ = %.2f]' % (x, y, th, v, kap) + title_adj)
    elif problem.env_name == 'PlanarQuad-v0':
        ax.set_title(title_adj)

    ax.legend(loc='best', framealpha=1.0)

    if figfile is None:
        plt.show()
    else:
        fig.savefig(figfile + '.pdf', dpi=DPI, transparent=True)
        plt.close(fig)
    print('visualize_eval end', flush=True);


def visualize_rollouts(starts, policy, problem,
                       figfile=None):
    print('visualize_rollouts', flush=True)

    if problem.env_name == 'DrivingOrigin-v0':
        x_idx = 0
        y_idx = 1
    elif problem.env_name == 'PlanarQuad-v0':
        x_idx = 0
        y_idx = 2

    fig, ax = plt.subplots()
    for start in starts:
        success, traj, actions = rl_utils.rollout(policy, start, problem, return_actions=True)

        x_arr = traj[:, x_idx]
        y_arr = traj[:, y_idx]
        # I don't do this now, but we could also
        # plot these w.r.t. time.
        # t_arr = traj[:, 2]
        # v_arr = traj[:, 3]
        # w_arr = traj[:, 4]

        if success:
            ax.plot(x_arr, y_arr, 'g')
        else:
            ax.plot(x_arr, y_arr, 'r')

    if problem.env_name == 'PlanarQuad-v0':
        for xo,yo,ro in zip(problem.env.unwrapped.obst_X, problem.env.unwrapped.obst_Y, problem.env.unwrapped.obst_R):
            c = plt.Circle((xo,yo),ro, color='black', alpha=1.0)
            ax.add_artist(c)

    green_line = mlines.Line2D([], [], color='g', label='Successful')
    red_line = mlines.Line2D([], [], color='r', label='Failed')
    if problem.env_name == 'DrivingOrigin-v0':
        ax.legend(handles=[green_line, red_line], loc='best')
    elif problem.env_name == 'PlanarQuad-v0':
        black_circle = mlines.Line2D([], [], color='white', marker='o', markerfacecolor="k", label='Obstacle')
        ax.legend(handles=[green_line, red_line, black_circle], loc='best')

    ax.set_title('Rollouts')
    ax.axis('equal')
    ax.set_xlabel('$x$ position')
    ax.set_ylabel('$y$ position')

    if figfile is None:
        plt.show()
    else:
        fig.savefig(figfile + '.pdf', dpi=DPI, transparent=True)
        plt.close(fig)

    # Plot state and action information too.
    if problem.env_name == 'PlanarQuad-v0':
        for coord in range(traj.shape[1]):
            fig, ax = plt.subplots()
            ax.plot(range(traj.shape[0]), traj[:, coord])
            fig.savefig(figfile + '_coord_%d.pdf' % coord, dpi=DPI, transparent=True)
            plt.close(fig)

        for coord in range(actions.shape[1]):
            fig, ax = plt.subplots()
            ax.plot(range(actions.shape[0]), actions[:, coord])
            fig.savefig(figfile + '_action_coord_%d.pdf' % coord, dpi=DPI, transparent=True)
            plt.close(fig)

    print('visualize_rollouts end', flush=True)

