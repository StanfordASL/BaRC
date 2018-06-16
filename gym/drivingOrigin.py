"""
Nonlinear kinematic car model implemented by 
James Harrison and Boris Ivanovic.

Implements a 5D state space where the agent drives to the origin.
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy
from scipy.integrate import odeint

logger = logging.getLogger(__name__)

class DrivingOriginEnv(gym.Env):
    """This implements the car model used in:
    "Kinodynamic RRT*: Optimal Motion Planning for Systems with Linear Differential Constraints"
    by Dustin Webb and Jur van den Berg
    https://arxiv.org/abs/1205.5088
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.dt = 0.1
        self.accel_limit = 2.
        self.kappa_limit = .5
        self.x_constraint = 3.#100.
        self.y_constraint = 3.#100.
        self.speed_constraint = 2.#3.
        self.theta_constraint = np.pi
        self.curvature_constraint = 0.1
        self.velocity_reward_coeff = 0.0
        self.control_coeff = 1.
        self.min_cost = -200.

        # What qualifies as a "success" such that we select it when expanding states?
        self.R_min = 0.5
        self.R_max = 1.0

        self.goal_pos_threshold = 0.1 # How many m away can you be from the goal and still finish?
        self.goal_vel_threshold = 0.1 # How many m/s different can you be from the goal velocity and still finish?
        self.goal_theta_threshold = 0.1 # How many radians different can you be from the goal theta and still finish?

        high_state = np.array([self.x_constraint,
                              self.y_constraint,
                              self.theta_constraint,
                              self.speed_constraint,
                              self.curvature_constraint])

        high_obsv = np.array([self.x_constraint,
                               self.y_constraint,
                               1.,
                               1.,
                               self.speed_constraint,
                               self.curvature_constraint])

        high_actions = np.array([self.accel_limit, self.kappa_limit])

        self.action_space = spaces.Box(low=-high_actions,high=high_actions)
        self.state_space = spaces.Box(low=-high_state, high=high_state)
        self.observation_space = spaces.Box(low=-high_obsv, high=high_obsv)
        self.seed(2015)
        self.viewer = None

        # Goal
        goal_v = self.np_random.uniform(low=0.1, high=1.)
        goal_theta = self.np_random.uniform(low=-np.pi, high=np.pi)

        # Setting a final state with zero curvature should be fine, this is
        # only used for sampling "nearby" states anyways. Any value could
        # be put instead of the 0. at the end.
        self.goal_state = np.array([0., 0., goal_theta, goal_v, 0.])
        
    def set_disturbance(self, disturbance_str):
        self.use_control_noise = False 
        self.use_nonzero_control_noise = False
        self.use_oversteer = False
        self.use_additive_velocity_noise = False

        if disturbance_str == 'control_noise':
            self.use_control_noise = True
            self.control_noise_source = np.random.RandomState(2015)
        elif disturbance_str == 'nonzero_control_noise':
            self.use_nonzero_control_noise = True
            self.nonzero_control_mean = 0.3
            self.nonzero_control_noise_source = np.random.RandomState(2015)
        elif disturbance_str == 'oversteer':
            self.use_oversteer = True
            self.oversteer_coeff = 1.5
        elif disturbance_str == 'velocity_noise':
            self.use_additive_velocity_noise = True
            self.velocity_noise_source = np.random.RandomState(2015)

        assert sum([self.use_control_noise, self.use_nonzero_control_noise, self.use_oversteer, self.use_additive_velocity_noise]) == 1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def x_dot(self,z,u):
        x,y,th,v,kap = z
        uv, uk = u
        x_d = [v*np.cos(th),v*np.sin(th),v*kap,uv,uk]
        return x_d

    def _get_obs(self, state):
        x,y,th,v,kap = state
        return np.array([x,y,np.cos(th),np.sin(th),v,kap])

    def step(self, action):
        old_state = self.state

        if self.use_control_noise:
            action += self.control_noise_source.normal(loc=0.0, scale=0.05, size=action.shape)
        elif self.use_nonzero_control_noise:
            action += self.nonzero_control_noise_source.normal(loc=self.nonzero_control_mean, scale=0.2, size=action.shape)
        elif self.use_oversteer:
            action[1] *= self.oversteer_coeff
        elif self.use_additive_velocity_noise:
            added_velocity = self.velocity_noise_source.normal(loc=0.0, scale=1.0)
            old_state[3] += added_velocity

        t = np.arange(0, self.dt, self.dt*0.01)
        integrand = lambda x,t: self.x_dot(old_state, action)
        x_tp1 = odeint(integrand, old_state, t)

        self.state = x_tp1[-1,:]
        if self.use_additive_velocity_noise:
            self.state[3] -= added_velocity
            old_state[3] -= added_velocity

        # Be close to the goal and have the desired final velocity.
        new_x, new_y, new_theta, new_v = self.state[:4]
        signed_delta_angle = np.arctan2(np.sin(new_theta - self.goal_state[2]), np.cos(new_theta - self.goal_state[2]))
        done = ((np.sqrt(new_x**2 + new_y**2) <= self.goal_pos_threshold) 
                and (np.abs(new_v - self.goal_state[3]) <= self.goal_vel_threshold) 
                and (np.abs(signed_delta_angle) <= self.goal_theta_threshold))

        # calculate reward; for now, just using euclidean reward
        # old_x, old_y, old_theta, old_v = old_state[:4]
        # uv, uk = action
        # reward = -(old_x**2 + old_y**2 
        #            + self.control_coeff*(uv**2 + uk**2) 
        #            + self.velocity_reward_coeff*old_v**2) # negative because we maximize
        
        reward = 1.0 if done else 0.0
        # reward = 0.0 if done else 1.0/self.min_cost

        return self._get_obs(self.state), reward, done, {}

    def reset(self):
        self.state = np.zeros(5)
        self.state[:3] = self.state_space.sample()[:3]
        self.state[3] = 0.001 # not controllable at v=0
        
        return self._get_obs(self.state)

    def render(self, mode='human', close=False):
        pass
