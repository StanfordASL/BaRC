import matlab.engine
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

X_IDX = 0
VX_IDX = 1
Y_IDX = 2
VY_IDX = 3
PHI_IDX = 4
W_IDX = 5

T1_IDX = 0
T2_IDX = 1

class Quad6DBackreachEngine:
    # Starts and sets up the MATLAB engine that runs in the background.
    def __init__(self):
        self.eng = matlab.engine.start_matlab();
        self.eng.cd("/home/borisi/boris-jam-backreach/code/backreach", nargout=0);
        self.eng.eval("addpath(genpath('/home/borisi/matlab-ToolboxLS/Kernel'));", nargout=0);
        self.eng.eval("addpath(genpath('/home/borisi/matlab-helperOC'));", nargout=0);
        

    # Setting up variables that will be used in subsequent calls
    def reset_variables(self, problem, plot_dir, 
                        tMax=0.1, nPoints=251.):
        self.eng.eval("global gX gY gW gVxPhi gVyPhi;", nargout=0);

        self.sampled_points = None
        self.actual_boundary = None
        self.most_recent_br_sets = None
        
        self.problem = problem
        self.plot_dir = plot_dir

        self.wRange = matlab.double([[self.problem.state_space.low[W_IDX]], [self.problem.state_space.high[W_IDX]]]);
        self.vxRange = matlab.double([[self.problem.state_space.low[VX_IDX]], [self.problem.state_space.high[VX_IDX]]]);
        self.vyRange = matlab.double([[self.problem.state_space.low[VY_IDX]], [self.problem.state_space.high[VY_IDX]]]);

        self.T1Min = self.T2Min = float(self.problem.env.unwrapped.Tmin)
        self.T1Max = self.T2Max = float(self.problem.env.unwrapped.Tmax)
        self.tMax = tMax;

        x_low, vx_low, y_low, vy_low, phi_low, w_low = self.problem.state_space.low;
        self.gMin = matlab.double([[x_low], [vx_low], [y_low], [vy_low], [phi_low], [w_low]]);

        x_high, vx_high, y_high, vy_high, phi_high, w_high = self.problem.state_space.high;
        self.gMax = matlab.double([[x_high], [vx_high], [y_high], [vy_high], [phi_high], [w_high]]);

        self.nPoints = nPoints
        self.gN = matlab.double((self.nPoints * np.ones((self.problem.state_dims, 1))).tolist());
        
        self.axis_coords = [np.linspace(self.gMin[i][0], self.gMax[i][0], nPoints) for i in range(self.problem.state_dims)]

        xg_lower = self.problem.env.unwrapped.xg_lower
        yg_lower = self.problem.env.unwrapped.yg_lower
        xg_upper = self.problem.env.unwrapped.xg_upper
        yg_upper = self.problem.env.unwrapped.yg_upper
        goal_w = self.problem.env.unwrapped.goal_w
        goal_vx = self.problem.env.unwrapped.goal_vx
        goal_vy = self.problem.env.unwrapped.goal_vy
        goal_phi = self.problem.env.unwrapped.goal_phi

        vRadius = self.problem.env.unwrapped.g_vel_limit
        phiRadius = self.problem.env.unwrapped.g_phi_limit
        posRadius = self.problem.env.unwrapped.g_pos_radius

        self.goalRectAndState = matlab.double([[xg_lower],
                                               [yg_lower], 
                                               [xg_upper], 
                                               [yg_upper],
                                               [goal_w],
                                               [goal_vx],
                                               [goal_vy],
                                               [goal_phi]]);

        self.goalRadii = matlab.double([[posRadius],
                                        [vRadius],
                                        [posRadius],
                                        [vRadius],
                                        [phiRadius],
                                        [vRadius]]);

        self.backreachRadii = matlab.double([[posRadius],
                                             [vRadius/2.],
                                             [posRadius],
                                             [vRadius/2.],
                                             [phiRadius/4.],
                                             [0.1]]);

        # This is the initial goal set. Note that we purposely keep 
        # the self.bestTarget__ variables as matlab types since we won't
        # be interacting with them ourselves (perhaps only for 
        # visualization).
        (self.bestTargetX, self.bestTargetY, self.bestTargetW, self.bestTargetVxPhi, self.bestTargetVyPhi) = \
            self.eng.Quad6D_create_init_target(self.gMin, 
                                                self.gMax, 
                                                self.gN,
                                                self.goalRectAndState,
                                                self.goalRadii,
                                                nargout=5);

        self.update_membership_functions(self.bestTargetX, self.bestTargetY, 
                                         self.bestTargetW, self.bestTargetVxPhi, 
                                         self.bestTargetVyPhi, 
                                         multi_data=False)


    def update_membership_functions(self, targetX, targetY, targetW, 
                                          targetVxPhi, targetVyPhi, 
                                          multi_data=True):
        np_tVxPhi = np.asarray(targetVxPhi)
        np_tVyPhi = np.asarray(targetVyPhi)

        np_tX = np.asarray(targetX)[:, -1]
        np_tY = np.asarray(targetY)[:, -1]
        np_tW = np.asarray(targetW)[:, -1]
        
        if multi_data:
            # For now, just take the last index, before updating this code to handle multiple returns
            self.numBRSreturns = np_tVxPhi.shape[-1]
            np_tVxPhi = np_tVxPhi[..., -1]
            np_tVyPhi = np_tVyPhi[..., -1]

        print('np_tVxPhi shape is', np_tVxPhi.shape, flush=True)
        print('np_tVyPhi shape is', np_tVyPhi.shape, flush=True)
        print('np_tX shape is', np_tX.shape, flush=True)
        print('np_tY shape is', np_tY.shape, flush=True)
        print('np_tW shape is', np_tW.shape, flush=True)
        
        # Here we update the membership-checking splines.
        self.VxPhi_check = RectBivariateSpline(x=self.axis_coords[VX_IDX], 
                                               y=self.axis_coords[PHI_IDX],
                                               z=np_tVxPhi,
                                               kx=1, ky=1)
        self.VyPhi_check = RectBivariateSpline(x=self.axis_coords[VY_IDX], 
                                               y=self.axis_coords[PHI_IDX],
                                               z=np_tVyPhi,
                                               kx=1, ky=1)
        self.x_check = interp1d(x=self.axis_coords[X_IDX], 
                                y=np_tX)
        self.y_check = interp1d(x=self.axis_coords[Y_IDX], 
                                y=np_tY)
        self.w_check = interp1d(x=self.axis_coords[W_IDX], 
                                y=np_tW)


    def any_start_in_goal(self, starts):
        for start in starts:
            if self.problem.env.unwrapped._in_goal(start):
                return True

        return False


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
        if self.any_start_in_goal(newStarts):
            # We do this so that we have the whole goal region if any 
            # of the starts end up being in the goal region.
            (self.bestTargetX, self.bestTargetY, self.bestTargetW, self.bestTargetVxPhi, self.bestTargetVyPhi) = \
            self.eng.Quad6D_create_init_target(self.gMin, 
                                                self.gMax, 
                                                self.gN,
                                                self.goalRectAndState,
                                                self.goalRadii,
                                                nargout=5);

        else:
            # If none are in the goal, then we just proceed with the cylindrical process
            # as usual.
            (self.bestTargetX, self.bestTargetY, self.bestTargetW, self.bestTargetVxPhi, self.bestTargetVyPhi) = \
                self.eng.Quad6D_create_update_target(self.gMin, 
                                                    self.gMax, 
                                                    self.gN,
                                                    matlab.double(newStarts[0].tolist()),
                                                    self.backreachRadii,
                                                    nargout=5);

        for start in newStarts[1:]:
            # To add cylinders to existing data grids, you must union them. This is why we keep the original grids.
            print('Quad6D_update_targets', flush=True);
            (self.bestTargetX, self.bestTargetY, self.bestTargetW, self.bestTargetVxPhi, self.bestTargetVyPhi) = \
                self.eng.Quad6D_update_targets(matlab.double(start.tolist()), 
                                                self.backreachRadii, 
                                                self.bestTargetX, 
                                                self.bestTargetY, 
                                                self.bestTargetW, 
                                                self.bestTargetVxPhi, 
                                                self.bestTargetVyPhi,
                                                nargout=5);
            print('Quad6D_update_targets end', flush=True);

        # These are the new values.
        print('Quad6D_approx_RS', flush=True);
        (targetX, targetY, targetW, targetVxPhi, targetVyPhi) = \
            self.eng.Quad6D_approx_RS( self.gMin,
                                       self.gMax,
                                       self.gN,
                                       self.T1Min,
                                       self.T1Max,
                                       self.T2Min,
                                       self.T2Max,
                                       self.wRange,
                                       self.vxRange,
                                       self.vyRange,
                                       self.bestTargetX,
                                       self.bestTargetY,
                                       self.bestTargetW,
                                       self.bestTargetVxPhi,
                                       self.bestTargetVyPhi,
                                       self.tMax,
                                       nargout=5);
        print('Quad6D_approx_RS end', flush=True);

        self.most_recent_br_sets = (targetX, targetY, targetW, targetVxPhi, targetVyPhi)
        self.update_membership_functions(targetX, targetY, targetW, targetVxPhi, targetVyPhi)
        self.update_contour_bounds(targetX, targetY, targetW, targetVxPhi, targetVyPhi)

        print('update_and_compute_backward_reachable_set end', flush=True);


    def evaluate_value_function(self, states):
        return (self.VxPhi_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False)
                + self.VyPhi_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False)
                + self.x_check(states[:, X_IDX])
                + self.y_check(states[:, Y_IDX])
                + self.w_check(states[:, W_IDX]))


    def check_membership(self, states):
        vxphi_membership = (self.VxPhi_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False) < 0).astype(bool)
        vyphi_membership = (self.VyPhi_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False) < 0).astype(bool)
        x_membership = (self.x_check(states[:, X_IDX]) < 0).astype(bool)
        y_membership = (self.y_check(states[:, Y_IDX]) < 0).astype(bool)
        w_membership = (self.w_check(states[:, W_IDX]) < 0).astype(bool)
        
        not_collision = np.zeros_like(states[:, 0]).astype(bool)
        for idx in range(states.shape[0]):
            not_collision[idx] = not self.problem.env.unwrapped._in_obst(states[idx])

        return np.logical_and(np.logical_and(np.logical_and(vxphi_membership, vyphi_membership),
                                             np.logical_and(x_membership, y_membership)),
                              np.logical_and(w_membership, not_collision))


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


    def update_contour_bounds(self, targetX, targetY, targetW, targetVxPhi, targetVyPhi):
        # Remember that MATLAB uses 1-based indexing whereas Python
        # uses 0-based indexing.
        max_vx, min_vx, max_phi1, min_phi1 = self.get_contour_bounds('gVxPhi', targetVxPhi, self.numBRSreturns);
        max_vy, min_vy, max_phi2, min_phi2 = self.get_contour_bounds('gVyPhi', targetVyPhi, self.numBRSreturns);
        max_phi = max(max_phi1, max_phi2);
        min_phi = min(min_phi1, min_phi2);

        max_x, min_x = self.get_plot_bounds(self.axis_coords[X_IDX], targetX);
        max_y, min_y = self.get_plot_bounds(self.axis_coords[Y_IDX], targetY);
        max_w, min_w = self.get_plot_bounds(self.axis_coords[W_IDX], targetW);

        self.contour_bounds = np.array([[min_x, min_vx, min_y, min_vy, min_phi, min_w], 
                                        [max_x, max_vx, max_y, max_vy, max_phi, max_w]]);
        print('[X, VX, Y, VY, PHI, W] bounds are', self.contour_bounds, flush=True);


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
            
            not_collision = np.zeros_like(potential_samples[:, 0]).astype(bool)
            for idx in range(potential_samples.shape[0]):
                not_collision[idx] = not self.problem.env.unwrapped._in_obst(potential_samples[idx])
            potential_samples = potential_samples[not_collision == True]

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
        grids = ['gX', 'gY', 'gW', 'gVxPhi', 'gVyPhi'];

        if value_funcs is None:
            value_funcs = [target for target in self.most_recent_br_sets];

        if grid_titles is None:
            grid_titles = ['BRS_X', 'BRS_Y', 'BRS_W', 'BRS_VxPhi', 'BRS_VyPhi'];
       
        if old_value_funcs is None:
            old_value_funcs = [self.bestTargetX, self.bestTargetY, self.bestTargetW, self.bestTargetVxPhi, self.bestTargetVyPhi]

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
                xlabel = grid[1:3];
                ylabel = grid[3:];

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

            if grid == 'gVxPhi':
                self.eng.eval("rectangle('Position', [%f, %f, %f, %f])" % (plot_boundary[0, VX_IDX], 
                                                                           plot_boundary[0, PHI_IDX], 
                                                                           plot_boundary[1, VX_IDX] - plot_boundary[0, VX_IDX], 
                                                                           plot_boundary[1, PHI_IDX] - plot_boundary[0, PHI_IDX]), nargout=0);
                legend_items.append(("Contour Bounding Box", 'k'));

                if self.sampled_points is not None:
                    vx_scatter = matlab.double(self.sampled_points[:, VX_IDX].tolist())
                    phi_scatter = matlab.double(self.sampled_points[:, PHI_IDX].tolist())
                    scatter_size = 5
                    self.eng.scatter(vx_scatter, phi_scatter, scatter_size, 'green', 'filled', nargout=0)
                    legend_items.append(("Sampled Points", 'fog'));

            elif grid == 'gVyPhi':
                self.eng.eval("rectangle('Position', [%f, %f, %f, %f])" % (plot_boundary[0, VY_IDX], 
                                                                           plot_boundary[0, PHI_IDX], 
                                                                           plot_boundary[1, VY_IDX] - plot_boundary[0, VY_IDX], 
                                                                           plot_boundary[1, PHI_IDX] - plot_boundary[0, PHI_IDX]), nargout=0);
                legend_items.append(("Contour Bounding Box", 'k'));

                if self.sampled_points is not None:
                    vy_scatter = matlab.double(self.sampled_points[:, VY_IDX].tolist())
                    phi_scatter = matlab.double(self.sampled_points[:, PHI_IDX].tolist())
                    scatter_size = 5
                    self.eng.scatter(vy_scatter, phi_scatter, scatter_size, 'green', 'filled', nargout=0)
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
