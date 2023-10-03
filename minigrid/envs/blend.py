from __future__ import annotations

import numpy as np

from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs.crossing import CrossingEnv
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.mission import MissionSpace

class BlendedEnv(MiniGridEnv):
    def __init__(self, t=1000, size=5, max_steps=None, **kwargs):
        # global blended_instance
        # blended_instance = self
        # Create a list of environments
        self.envs = [
            CrossingEnv(size=size, max_steps=max_steps, **kwargs),
            DoorKeyEnv(size=size, max_steps=max_steps, **kwargs)
        ]

        # Which environment is currently active
        self.current_env_idx = 0
        self.current_env = self.envs[self.current_env_idx]
        self.episode_count = 0
        self.t = t
        mission_space = MissionSpace(mission_func=lambda: self.current_env._gen_mission())

        super().__init__(grid_size=size, mission_space=mission_space, **kwargs)
        self.step_count = 0  # Initialize step count for blended environment



    def step(self, action):
        # Perform the step in the current environment
        self.step_count += 1  # Increment step count
        obs, reward, done, truncated, info = self.current_env.step(action)

        if done:
            return obs, reward, done, truncated, info

        # If it's time to swap the environment
        if self.step_count % self.t == 0:
            obs = self.swap_env(obs)

        return obs, reward, done, truncated, info

    def swap_env(self, obs):
        # Get the current agent position from the observation's image
        agent_identifier = [1, 0, 0]
        agent_positions = np.argwhere(np.all(obs['image'] == agent_identifier, axis=-1))

        if agent_positions.shape[0] > 0:
            agent_pos = agent_positions[0]
        else:
            agent_pos = [0, 0]  # Default value in case the agent's position isn't found

        # Switch the environment
        self.current_env_idx = 1 - self.current_env_idx
        self.current_env = self.envs[self.current_env_idx]

        # Adjust agent_pos to be within the new environment's valid navigable area
        agent_pos[0] = min(max(agent_pos[0], 1), self.current_env.width - 2)
        agent_pos[1] = min(max(agent_pos[1], 1), self.current_env.height - 2)

        # Reset the new environment and place the agent in the same position
        new_obs = self.current_env.reset()
        self.current_env.agent_pos = agent_pos
        self.current_env.agent_dir = obs['direction']  # Assuming the direction is the same
        # Regenerate the observation after changing the agent's position and direction
        new_obs = self.current_env.gen_obs()
        return new_obs

    def reset(self, **kwargs):
        self.episode_count += 1
        if self.episode_count % self.t == 0:
            self.swap_active_env()
        obs = self.current_env.reset(**kwargs)
        return obs




    def render(self, *args, **kwargs):
        return self.current_env.render(*args, **kwargs)

    def _gen_mission(self):
        return self.current_env._gen_mission()
