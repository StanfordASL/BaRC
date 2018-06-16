import matlab.engine
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

X_IDX = 0
Y_IDX = 1
THETA_IDX = 2
V_IDX = 3
W_IDX = 4

ACCEL_IDX = 0
KAPPA_IDX = 1

class Car5DBackreachEngine:
    # Starts and sets up the MATLAB engine that runs in the background.
    def __init__(self):
        self.eng = matlab.engine.start_matlab();
        self.eng.cd("/home/borisi/boris-jam-backreach/code/backreach", nargout=0);
        self.eng.eval("addpath(genpath('/home/borisi/matlab-ToolboxLS/Kernel'));", nargout=0);
        self.eng.eval("addpath(genpath('/home/borisi/matlab-helperOC'));", nargout=0);
        

    # Setting up variables that will be used in subsequent calls
    def reset_variables(self, problem, plot_dir, 
                        tMax=0.1, nPoints=251., 
                        wRadius=0.1): # Kind of arbitrary...
        self.eng.eval("global gXT gYT gV gW;", nargout=0);

        self.sampled_points = None
        self.actual_boundary = None
        self.most_recent_br_sets = None
        
        self.problem = problem
        self.plot_dir = plot_dir

        self.vRange = matlab.double([[0.0], [self.problem.state_space.high[V_IDX]]]);
        self.aMax = float(self.problem.action_space.high[ACCEL_IDX]);
        self.alphaMax = float(self.problem.action_space.high[KAPPA_IDX]);
        # max theta_dot = max velocity * max kappa
        self.wMax = float(self.problem.state_space.high[V_IDX] * self.alphaMax);
        self.tMax = tMax;

        x_low, y_low, theta_low, v_low, curv_low = self.problem.state_space.low;
        self.gMin = matlab.double([[x_low], [y_low], [theta_low], [v_low], [curv_low]]);

        x_high, y_high, theta_high, v_high, curv_high = self.problem.state_space.high;
        self.gMax = matlab.double([[x_high], [y_high], [theta_high], [v_high], [curv_high]]);

        self.nPoints = nPoints
        self.gN = matlab.double((self.nPoints * np.ones((self.problem.state_dims, 1))).tolist());
        
        self.axis_coords = [np.linspace(self.gMin[i][0], self.gMax[i][0], nPoints) for i in range(self.problem.state_dims)]

        goalX, goalY, goalTheta, goalV, goalW = self.problem.goal_state
        xRadius = yRadius = self.problem.env.unwrapped.goal_pos_threshold
        thetaRadius = self.problem.env.unwrapped.goal_theta_threshold
        vRadius = self.problem.env.unwrapped.goal_vel_threshold

        # Since x and theta and y are coupled in the decomposition,
        # we will also couple their radii.
        thetaRadius = xRadius = yRadius = np.max([xRadius, thetaRadius]);

        self.goalState = matlab.double([[goalX],
                                       [goalY], 
                                       [goalTheta], 
                                       [goalV],
                                       [goalW]]);

        self.goalRadii = matlab.double([[xRadius], 
                                       [yRadius], 
                                       [thetaRadius], 
                                       [vRadius], 
                                       [wRadius]]);

        # This is the initial goal set. Note that we purposely keep 
        # the self.bestTarget__ variables as matlab types since we won't
        # be interacting with them ourselves (perhaps only for 
        # visualization).
        (self.bestTargetXT, self.bestTargetYT, self.bestTargetV, self.bestTargetW) = \
            self.eng.Plane5D_create_init_target(self.gMin, 
                                                self.gMax, 
                                                self.gN,
                                                self.goalState,
                                                self.goalRadii,
                                                nargout=4);

        self.update_membership_functions(self.bestTargetXT, self.bestTargetYT, 
                                         self.bestTargetV,  self.bestTargetW, 
                                         multi_data=False)


    def update_membership_functions(self, targetXT, targetYT, targetV, targetW, multi_data=True):
        np_tXT = np.asarray(targetXT)
        np_tYT = np.asarray(targetYT)
        np_tV = np.asarray(targetV)[:, -1]
        np_tW = np.asarray(targetW)[:, -1]
        
        if multi_data:
            # For now, just take the last index, before updating this code to handle multiple returns
            self.numBRSreturns = np_tXT.shape[-1]
            np_tXT = np_tXT[..., -1]
            np_tYT = np_tYT[..., -1]

        print('np_tXT shape is', np_tXT.shape, flush=True)
        print('np_tYT shape is', np_tYT.shape, flush=True)
        print('np_tV shape is', np_tV.shape, flush=True)
        print('np_tW shape is', np_tW.shape, flush=True)
        
        # Here we update the membership-checking splines.
        self.xt_check = RectBivariateSpline(x=self.axis_coords[X_IDX], 
                                            y=self.axis_coords[THETA_IDX],
                                            z=np_tXT,
                                            kx=1, ky=1)
        self.yt_check = RectBivariateSpline(x=self.axis_coords[Y_IDX], 
                                            y=self.axis_coords[THETA_IDX],
                                            z=np_tYT,
                                            kx=1, ky=1)
        self.v_check = interp1d(x=self.axis_coords[V_IDX], 
                                y=np_tV)
        self.w_check = interp1d(x=self.axis_coords[W_IDX], 
                                y=np_tW)


    # Acts as an interface to the MATLAB/C++ code.
    def update_and_compute_backward_reachable_set(self, newStarts, 
                                                  plot=False,
                                                  curr_train_iter=0):
        print('update_and_compute_backward_reachable_set', flush=True);
        if len(newStarts) == 0:
            print('len(newStarts) == 0, returning!', flush=True);
            return

        # Creating a new target which is only comprised of new starts, 
        # rather than previously keeping all new starts that we ever looked at.
        (self.bestTargetXT, self.bestTargetYT, self.bestTargetV, self.bestTargetW) = \
            self.eng.Plane5D_create_init_target(self.gMin, 
                                                self.gMax, 
                                                self.gN,
                                                matlab.double(newStarts[0].tolist()),
                                                self.goalRadii,
                                                nargout=4);

        for start in newStarts[1:]:
            # To add cylinders to existing data grids, you must union them. This is why we keep the original grids.
            print('Plane5D_update_targets', flush=True);
            (self.bestTargetXT, self.bestTargetYT, self.bestTargetV, self.bestTargetW) = \
                self.eng.Plane5D_update_targets(matlab.double(start.tolist()), 
                                                self.goalRadii, 
                                                self.bestTargetXT, 
                                                self.bestTargetYT, 
                                                self.bestTargetV, 
                                                self.bestTargetW, 
                                                nargout=4);
            print('Plane5D_update_targets end', flush=True);

        # These are the new values.
        print('Plane5D_approx_RS', flush=True);
        (targetXT, targetYT, targetV, targetW) = \
            self.eng.Plane5D_approx_RS(self.gMin, 
                                       self.gMax, 
                                       self.gN, 
                                       self.aMax,
                                       self.alphaMax, 
                                       self.wMax, 
                                       self.vRange,
                                       self.tMax, 
                                       self.bestTargetXT, 
                                       self.bestTargetYT, 
                                       self.bestTargetV, 
                                       self.bestTargetW, 
                                       nargout=4);
        print('Plane5D_approx_RS end', flush=True);

        self.most_recent_br_sets = (targetXT, targetYT, targetV, targetW)
        self.update_membership_functions(targetXT, targetYT, targetV, targetW)
        self.update_contour_bounds(targetXT, targetYT, targetV, targetW)

        print('update_and_compute_backward_reachable_set end', flush=True);


    def evaluate_value_function(self, states):
        return (self.xt_check(states[:, X_IDX], states[:, THETA_IDX], grid=False)
                + self.yt_check(states[:, Y_IDX], states[:, THETA_IDX], grid=False)
                + self.v_check(states[:, V_IDX])
                + self.w_check(states[:, W_IDX]))


    def check_membership(self, states):
        xt_membership = (self.xt_check(states[:, X_IDX], states[:, THETA_IDX], grid=False) < 0).astype(bool)
        yt_membership = (self.yt_check(states[:, Y_IDX], states[:, THETA_IDX], grid=False) < 0).astype(bool)
        v_membership = (self.v_check(states[:, V_IDX]) < 0).astype(bool)
        w_membership = (self.w_check(states[:, W_IDX]) < 0).astype(bool)
        return np.logical_and(np.logical_and(xt_membership, yt_membership),
                              np.logical_and(v_membership, w_membership))


    def remove_indicator_values(self, contour_matrix):
        C = list();
        idx = 0;
        next_idx = 0;
        while next_idx < contour_matrix.shape[1]:
            num_elements = contour_matrix[1, idx]
            next_idx = int(idx + 1 + num_elements)
            C.append(contour_matrix[:, idx+1:next_idx])
            idx = next_idx

        return np.concatenate(C, axis=1)


    def get_contour_bounds(self, grid_name, data, brs_idx):
        self.eng.workspace['BRS_data'] = data;
        self.eng.eval('C = getContour(%s, BRS_data, %d);' % (grid_name, brs_idx), nargout=0);
        C = np.asarray(self.eng.workspace['C']);
        C = self.remove_indicator_values(C)
        
        max_x = C[0].max();
        min_x = C[0].min();
        max_y = C[1].max();
        min_y = C[1].min();
        return max_x, min_x, max_y, min_y


    def get_plot_bounds(self, axis_points, data):
        # For now, just take the last index, before updating this code to handle multiple returns
        data_arr = np.asarray(data)[..., -1];
        
        assert axis_points.size == data_arr.size

        min_idx = min(idx for idx, v in enumerate(data_arr) if v < 0)
        if min_idx > 0:
            min_idx -= 1;
        min_v = axis_points[min_idx];

        max_idx = max(idx for idx, v in enumerate(data_arr) if v < 0)
        if max_idx < data_arr.size - 1:
            max_idx += 1;
        max_v = axis_points[max_idx];

        return max_v, min_v


    def update_contour_bounds(self, targetXT, targetYT, targetV, targetW):
        # Remember that MATLAB uses 1-based indexing whereas Python
        # uses 0-based indexing.
        max_x, min_x, max_theta1, min_theta1 = self.get_contour_bounds('gXT', targetXT, self.numBRSreturns);
        max_y, min_y, max_theta2, min_theta2 = self.get_contour_bounds('gYT', targetYT, self.numBRSreturns);
        max_theta = max(max_theta1, max_theta2);
        min_theta = min(min_theta1, min_theta2);

        max_v, min_v = self.get_plot_bounds(self.axis_coords[V_IDX], targetV);
        max_w, min_w = self.get_plot_bounds(self.axis_coords[W_IDX], targetW);

        self.contour_bounds = np.array([[min_x, min_y, min_theta, min_v, min_w], 
                                        [max_x, max_y, max_theta, max_v, max_w]]);
        print('[X, Y, T, V, W] bounds are', self.contour_bounds, flush=True);


    def sample_from_grid(self, size=1, method='uniform'):
        print('sample_from_grid, method =', method, flush=True);
        if method == 'uniform':
            self.actual_boundary = self.contour_bounds

            num_successfully_sampled = 0;
            successful_samples = list()
            while num_successfully_sampled < size:
                print('sample_from_grid while loop', flush=True);
                
                potential_samples = np.random.uniform(low=self.actual_boundary[0], high=self.actual_boundary[1], size=(size, self.problem.state_dims))
                membership = self.check_membership(potential_samples)
                member_samples = potential_samples[membership == True]
                num_sampled = member_samples.shape[0]
                if num_sampled == 0:
                    continue

                num_successfully_sampled += num_sampled
                successful_samples.append(member_samples)

            self.sampled_points = np.concatenate(successful_samples);
            print('sample_from_grid done', flush=True);
            return self.sampled_points

        elif method == 'contour_edges':
            # Start by sampling 5x the points from a 20% larger box
            pct_added = 0.20
            self.actual_boundary = np.zeros((2, self.problem.state_dims))
            self.actual_boundary[0] = np.maximum(self.problem.state_space.low, self.contour_bounds[0] - pct_added*np.abs(self.contour_bounds[0]))
            self.actual_boundary[1] = np.minimum(self.problem.state_space.high, self.contour_bounds[1] + pct_added*np.abs(self.contour_bounds[1]))

            print("actual_boundary")
            print(self.actual_boundary)
            print("contour_bounds")
            print(self.contour_bounds)
            potential_samples = np.random.uniform(low=self.actual_boundary[0], 
                                                  high=self.actual_boundary[1], 
                                                  size=(size*5, self.problem.state_dims))
            
            # Then, evaluate their sampling weights
            weights = np.abs(self.evaluate_value_function(potential_samples))
            weights = np.reciprocal(weights)
            weights /= np.sum(weights)

            sampled_idxs = np.random.choice(potential_samples.shape[0], 
                                            size=size,
                                            p=weights)
            
            self.sampled_points = potential_samples[sampled_idxs]
            print('sample_from_grid done', flush=True);
            return self.sampled_points

        else:
            raise ValueError('Unknown grid sampling method:', method);


    def visualize_grids(self, 
                        file_prefix='', 
                        file_suffix='', 
                        value_funcs=None, 
                        old_value_funcs=None, 
                        grid_titles=None):
        print('visualize_grids', flush=True);
        grids = ['gXT', 'gYT', 'gV', 'gW'];

        if value_funcs is None:
            value_funcs = [target for target in self.most_recent_br_sets];

        if grid_titles is None:
            grid_titles = ['BRS_XT', 'BRS_YT', 'BRS_V', 'BRS_W'];
       
        if old_value_funcs is None:
            old_value_funcs = [self.bestTargetXT, self.bestTargetYT, self.bestTargetV, self.bestTargetW]

        if old_value_funcs is not None:
            first_color = 'red'
            second_color = 'blue'
        else:
            first_color = 'blue'

        for idx, grid in enumerate(grids):
            print('visualize_grids %s' % grid, flush=True);
            self.eng.eval("gdim = %s.dim;" % grid, nargout=0);
            gdim = int(self.eng.workspace['gdim']);
            if gdim == 1:
                xlabel = grid[1];
                ylabel = '';
            elif gdim == 2:
                xlabel = grid[1];
                ylabel = grid[2];

            legend_items = list();
            
            self.eng.eval("fig = figure;", nargout=0);
            self.eng.workspace['value_func'] = value_funcs[idx];
            self.eng.eval("extra.deleteLastPlot = false;", nargout=0);
            self.eng.eval("visSetIm(%s, value_func, '%s', 0, extra); hold on;" % (grid, first_color), nargout=0);

            if old_value_funcs is None:
                legend_items.append(("Target Set", 'b'));
            else:
                legend_items.append(("BR Set", 'r'));
                if gdim == 1:
                    legend_items.append(("Membership Boundary", ':k'));

                self.eng.workspace['old_value_func'] = old_value_funcs[idx];
                self.eng.eval("visSetIm(%s, old_value_func, '%s', 0, extra); hold on;" % (grid, second_color), nargout=0);
                legend_items.append(("Target Set", 'b'));

            # contour_bounds looks like np.array([[min_x, min_y, min_theta, min_v, min_w], 
            #                                     [max_x, max_y, max_theta, max_v, max_w]]);
            if self.actual_boundary is None:
                plot_boundary = self.contour_bounds
            else:
                plot_boundary = self.actual_boundary

            if grid == 'gXT':
                self.eng.eval("rectangle('Position', [%f, %f, %f, %f])" % (plot_boundary[0, X_IDX], 
                                                                           plot_boundary[0, THETA_IDX], 
                                                                           plot_boundary[1, X_IDX] - plot_boundary[0, X_IDX], 
                                                                           plot_boundary[1, THETA_IDX] - plot_boundary[0, THETA_IDX]), nargout=0);
                legend_items.append(("Contour Bounding Box", 'k'));

                if self.sampled_points is not None:
                    x_scatter = matlab.double(self.sampled_points[:, X_IDX].tolist())
                    t_scatter = matlab.double(self.sampled_points[:, THETA_IDX].tolist())
                    scatter_size = 5
                    self.eng.scatter(x_scatter, t_scatter, scatter_size, 'green', 'filled', nargout=0)
                    legend_items.append(("Sampled Points", 'fog'));

            elif grid == 'gYT':
                self.eng.eval("rectangle('Position', [%f, %f, %f, %f])" % (plot_boundary[0, Y_IDX], 
                                                                           plot_boundary[0, THETA_IDX], 
                                                                           plot_boundary[1, Y_IDX] - plot_boundary[0, Y_IDX], 
                                                                           plot_boundary[1, THETA_IDX] - plot_boundary[0, THETA_IDX]), nargout=0);
                legend_items.append(("Contour Bounding Box", 'k'));

                if self.sampled_points is not None:
                    y_scatter = matlab.double(self.sampled_points[:, Y_IDX].tolist())
                    t_scatter = matlab.double(self.sampled_points[:, THETA_IDX].tolist())
                    scatter_size = 5
                    self.eng.scatter(y_scatter, t_scatter, scatter_size, 'green', 'filled', nargout=0)
                    legend_items.append(("Sampled Points", 'fog'));

            self.eng.eval("h = zeros(%d, 1);" % len(legend_items), nargout=0);
            names_list = list()
            for leg_idx, leg_item in enumerate(legend_items):
                if leg_item[1] == 'fog':
                    self.eng.eval("h(%d) = plot(NaN, NaN, 'o', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g');" % (leg_idx + 1), nargout=0);
                else:
                    self.eng.eval("h(%d) = plot(NaN, NaN, '%s');" % (leg_idx + 1, leg_item[1]), nargout=0);
                
                names_list.append("'%s'" % leg_item[0])

            filename = file_prefix + grid_titles[idx] + file_suffix
            self.eng.eval("title('%s'); xlabel('%s'); ylabel('%s'); tightfig;" % (grid_titles[idx].replace('_', ' ') + ' Level = 0', xlabel, ylabel), nargout=0);
            self.eng.eval("legend(h, %s, 'Location', 'best');" % ','.join(names_list), nargout=0);
            self.eng.eval("print(fig, '%s', '-dpdf', '-r300', '-noui');" % filename, nargout=0);
            self.eng.eval("close(fig)", nargout=0);

        print('visualize_grids end', flush=True);


    def stop(self):
        self.eng.quit()
        self.eng = None
