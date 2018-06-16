import gym
import numpy as np
from utils import weighted_sample, signed_delta_angle

class Problem(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_name, zero_goal_v=False, disturbance=None):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.env.unwrapped.set_disturbance(disturbance)

        if zero_goal_v:
            # Setting v to be very small (for plotting), but still
            # basically zero for RL purposes.
            self.env.unwrapped.goal_state[3] = 1e-4;

        # Making sure we're working with vector spaces 
        # (not matrices or higher, e.g. images, MRI volumes, etc)
        assert len(self.env.observation_space.shape) == 1

        self.observation_space = self.env.observation_space
        self.state_space = self.env.unwrapped.state_space
        self.action_space = self.env.action_space

        self.state_dims = self.state_space.shape[0]

        self.start_state_dist = None
        self.goal_state = self.env.unwrapped.goal_state
        self.min_cost = self.env.unwrapped.min_cost

        self.R_min = self.env.unwrapped.R_min
        self.R_max = self.env.unwrapped.R_max


    def sample_from_space(self, 
                          num_states=100, 
                          numpy_array=False, 
                          zero_idxs=list()):
        mask = np.ones((self.state_dims, ))
        for idx in zero_idxs:
            mask[idx] = 0.0

        ret = [self.state_space.sample()*mask for _ in range(num_states)]
        if numpy_array:
            return np.stack(ret)

        return ret


    def sample_behind_goal(self, goal_state,
                           num_states=100, 
                           numpy_array=False, 
                           zero_idxs=list(), 
                           angle_radius=np.pi/4.):
        samples = list()
        behind_goal_theta = goal_state[2] + np.pi

        while len(samples) < num_states:
            angles = np.random.uniform(low=behind_goal_theta - angle_radius,
                                       high=behind_goal_theta + angle_radius,
                                       size=(num_states, 1))
            radii = np.random.uniform(low=0.0, 
                                      high=self.observation_space.high[0]*np.sqrt(2.), 
                                      size=(num_states, 1))
            point_angles = np.random.uniform(low=goal_state[2] - angle_radius,
                                             high=goal_state[2] + angle_radius,
                                             size=(num_states, ))

            sampled_points = np.concatenate([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

            for idx, state in enumerate(sampled_points):
                if (state >= self.state_space.low[0:2]).all() and (state <= self.state_space.high[0:2]).all():
                    samples.append(np.array([state[0], state[1], point_angles[idx], 0., 0.]))

        if numpy_array:
            return np.stack(samples)

        return samples

    
    def set_state(self, new_state):
        self.env.unwrapped.state = new_state


    def act_from_state(self, state, action):
        self.set_state(state)
        return self.step(action)


    def step(self, action, ret_state=False):
        ob, rew, done, fourth_thing = self.env.step(action)
        if ret_state:
            ob = self.env.unwrapped.state

        return ob, rew, done, fourth_thing


    def reset_to_state(self, new_start_state):
        self.env.reset()
        self.set_state(new_start_state)
        return self.get_obs(new_start_state)


    def get_obs(self, x):
        return self.env.unwrapped._get_obs(x)


    def reset(self, ret_state_and_ob=False):
        return_ob = self.env.reset()

        if self.start_state_dist is not None:
            start_state = weighted_sample(self.start_state_dist)
            self.set_state(start_state)
            if ret_state_and_ob:
                return_ob = start_state, self.get_obs(start_state)
            else:
                return_ob = self.get_obs(start_state)

        return return_ob


    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)
