import matplotlib
matplotlib.use('Agg');

from matplotlib import rc
rc('font',**{'family':'serif'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerRegularPolyCollection

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
                     x_idx=0, y_idx=1, theta_idx=2, v_idx=3,
                     edgecolor='k', size=15, edgewidths=0.2,
                     goal_color='b', old_start_color='grey',
                     old_start_name='Old Starts', title_adj='',
                     figfile=None):
    goal_state = problem.goal_state

    fig, ax = plt.subplots()

    # Old Starts
    if old_starts is not None:
        old_xy = np.stack(old_starts)
        ax.scatter(old_xy[:, x_idx], old_xy[:, y_idx], 
                   c=old_start_color, edgecolor=edgecolor,
                   s=size, label=old_start_name,
                   linewidths=edgewidths, alpha=0.5)

    # Starts
    xy = np.stack(starts)
    if rewards_map is None:
        starts_color = 'g'
        cm = None
    else:
        cm = plt.cm.get_cmap('Greens')
        starts_color = list()
        for start in starts:
            start_str = start.tostring()
            starts_color.append(rewards_map[start_str] 
                                    if start_str in rewards_map else 0.0)

    sc = ax.scatter(xy[:, x_idx], xy[:, y_idx], 
                    c=starts_color, edgecolor=edgecolor,
                    s=size, label='Starts' if old_starts is None else 'New Starts',
                    cmap=cm, vmin=0, vmax=1,
                    linewidths=edgewidths, alpha=0.8)
    
    if cm is not None:
        fig.colorbar(sc)

    # Goal State
    ax.plot(goal_state[x_idx], goal_state[y_idx], 
            c=goal_color, marker='o', markeredgecolor=edgecolor, 
            markersize=np.sqrt(size), label='Goal', linestyle='None',
            markeredgewidth=edgewidths)
    ax.arrow(x=goal_state[x_idx], y=goal_state[y_idx], 
             dx=goal_state[v_idx]*np.cos(goal_state[theta_idx]), 
             dy=goal_state[v_idx]*np.sin(goal_state[theta_idx]),
             head_width=0.1, head_length=0.1, fc=goal_color, ec=goal_color,
             alpha=0.5)

    # ax.set_ylim(problem.observation_space.low[y_idx],  # Bottom
    #             problem.observation_space.high[y_idx]) # Top
    # ax.set_xlim(problem.observation_space.low[x_idx],  # Bottom
    #             problem.observation_space.high[x_idx]) # Top
    ax.axis('equal')
    ax.set_xlabel('$x$ position')
    ax.set_ylabel('$y$ position')

    x, y, th, v, kap = goal_state
    ax.set_title(r'Goal State: [$x$ = %.2f, $y$ = %.2f, $\theta$ = %.2f, $v$ = %.2f, $\kappa$ = %.2f]' % (x, y, th, v, kap) + title_adj)

    if cm is not None:
        legend_dict = dict(ncol=1, loc='best', scatterpoints=4, framealpha=1.0)
        ax.legend(handler_map={type(sc) : ScatterHandler()}, **legend_dict)
    else:
        ax.legend(loc='best', framealpha=1.0)

    if figfile is None:
        plt.show()
    else:
        fig.savefig(figfile + '.pdf', dpi=DPI, transparent=True)
        plt.close(fig)


def visualize_eval(starts, problem, results,
                   x_idx=0, y_idx=1, theta_idx=2, v_idx=3,
                   edgecolor='k', size=15, edgewidths=0.2,
                   goal_color='b', title_adj='',
                   figfile=None):
    goal_state = problem.goal_state

    fig, ax = plt.subplots()

    # Starts
    successful_xy = list()
    failed_xy = list()
    starts = np.stack(starts);
    for i in range(starts.shape[0]):
        if results[i] == 1.0:
            successful_xy.append(starts[i])
        else:
            failed_xy.append(starts[i])

    if len(successful_xy) > 0:
        successful_xy = np.stack(successful_xy)
    else:
        successful_xy = None

    x_plot = successful_xy[:, x_idx] if successful_xy is not None else list()
    y_plot = successful_xy[:, y_idx] if successful_xy is not None else list()
    ax.scatter(x_plot, y_plot, 
               c='g', edgecolor=edgecolor,
               s=size, label='Successful Starts',
               linewidths=edgewidths, alpha=0.8)
    
    if len(failed_xy) > 0:
        failed_xy = np.stack(failed_xy)
    else:
        failed_xy = None
    
    x_plot = failed_xy[:, x_idx] if failed_xy is not None else list()
    y_plot = failed_xy[:, y_idx] if failed_xy is not None else list()
    ax.scatter(x_plot, y_plot, 
               c='r', edgecolor=edgecolor,
               s=size, label='Failed Starts',
               linewidths=edgewidths, alpha=0.8)

    # Goal State
    ax.plot(goal_state[x_idx], goal_state[y_idx], 
            c=goal_color, marker='o', markeredgecolor=edgecolor, 
            markersize=np.sqrt(size), label='Goal', linestyle='None',
            markeredgewidth=edgewidths)
    ax.arrow(x=goal_state[x_idx], y=goal_state[y_idx], 
             dx=goal_state[v_idx]*np.cos(goal_state[theta_idx]), 
             dy=goal_state[v_idx]*np.sin(goal_state[theta_idx]),
             head_width=0.1, head_length=0.1, fc=goal_color, ec=goal_color,
             alpha=0.5)

    # ax.set_ylim(problem.observation_space.low[y_idx],  # Bottom
    #             problem.observation_space.high[y_idx]) # Top
    # ax.set_xlim(problem.observation_space.low[x_idx],  # Bottom
    #             problem.observation_space.high[x_idx]) # Top
    ax.axis('equal')
    ax.set_xlabel('$x$ position')
    ax.set_ylabel('$y$ position')

    x, y, th, v, kap = goal_state
    ax.set_title(r'Goal State: [$x$ = %.2f, $y$ = %.2f, $\theta$ = %.2f, $v$ = %.2f, $\kappa$ = %.2f]' % (x, y, th, v, kap) + title_adj)
    ax.legend(loc='best', framealpha=1.0)

    if figfile is None:
        plt.show()
    else:
        fig.savefig(figfile + '.pdf', dpi=DPI, transparent=True)
        plt.close(fig)
