from __future__ import annotations

import itertools as itt

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs.crossing import CrossingEnv
from minigrid.envs.doorkey import DoorKeyEnv

# from minigrid.utils.helper import blended_mission_helper

class BlendedEnv(MiniGridEnv):
    def __init__(self, t=5, seed=None, size=5, max_steps=None, render_mode=None, **kwargs):
        global blended_instance
        blended_instance = self
        mission_space = MissionSpace(mission_func=blended_mission_helper)

        super().__init__(grid_size=size, mission_space=mission_space, **kwargs)
        
        # Create a list of environments
        self.envs = [
            CrossingEnv(size=size, max_steps=max_steps, **kwargs),
            DoorKeyEnv(size=size, max_steps=max_steps, **kwargs)
        ]
        
        # Which environment is currently active
        self.current_env_idx = 0
        
        # Current active environment instance
        self.current_env = self.envs[self.current_env_idx]
        
        # Episode counter
        self.episode_count = 0
        
        # Number of episodes before switching
        self.t = t
        
        # Seed
        self.seed = seed

    def step(self, action):
        return self.current_env.step(action)
    def reset(self, *args, **kwargs):
        obs = self.current_env.reset(*args, **kwargs)
        
        # Update episode count
        self.episode_count += 1
        
        # If it's time to switch environments
        if self.episode_count % self.t == 0:
            self.current_env_idx = (self.current_env_idx + 1) % len(self.envs)
            self.current_env = self.envs[self.current_env_idx]
            
        return obs
    def render(self, *args, **kwargs):
        return self.current_env.render(*args, **kwargs)

    def _gen_mission(self):
        if hasattr(self, 'current_env') and self.current_env is not None:
            return self.current_env._gen_mission()
        else:
            return "Complete the task!"
    # Directly use the grid from the current env
    # def _gen_grid(self, width, height):
    #     return self.current_env._gen_grid(width, height)

    # def __init__(self, env_name_1='MiniGrid-DoorKey-6x6-v0', env_name_2='MiniGrid-GoToDoor-6x6-v0', t=5, seed=None, size=5, render_mode=None, **kwargs):
    #     def __init__(self, t=5, seed=None, size=5, render_mode=None, **kwargs):
    #     mission_space = MissionSpace(mission_func=self._gen_mission)
    #     super().__init__(mission_space=mission_space, grid_size=size, **kwargs)
        
    #     # Set the environment names
    #     self.env_names = [
    #         "DoorKeyEnv",      # Corresponding to MiniGrid-DoorKey-5x5-v0
    #         "CrossingEnv"      # Corresponding to MiniGrid-LavaCrossingS9N1-v0
    #     ]
        
    #     # We'll use the size argument to configure the DoorKeyEnv.
    #     # LavaCrossingS9N1 seems to be fixed size, so we won't use the size argument for it.
    #     self.env_args = [
    #         {"size": size, "render_mode": render_mode},
    #         {"render_mode": render_mode}
    #     ]
        
    #     # Which environment is currently active
    #     self.current_env_idx = 0
        
    #     # Current active environment instance
    #     self.current_env = None
        
    #     # Episode counter
    #     self.episode_count = 0
        
    #     # Number of episodes before switching
    #     self.t = t
        
    #     # Seed
    #     self.seed = seed
        
    # def reset(self):
    #     # Check if it's time to switch environments
    #     if self.episode_count > 0 and self.episode_count % self.t == 0:
    #         self.current_env_idx = (self.current_env_idx + 1) % len(self.env_names)
        
    #     env_name = self.env_names[self.current_env_idx]
    #     args = self.env_args[self.current_env_idx]
        
    #     self.current_env = globals()[env_name](**args)
        
    #     if self.seed is not None:
    #         self.current_env.seed(self.seed + self.current_env_idx)
        
    #     self.episode_count += 1
        
    #     return self.current_env.reset()

    # def step(self, action):
    #     return self.current_env.step(action)
    
    # def render(self, *args, **kwargs):
    #     return self.current_env.render(*args, **kwargs)

    # @staticmethod
    # def _gen_mission():
    #     return "do whatever"

    # def _gen_grid(self, width, height):
    #     pass

    # def __init__(self, t=5, size=8, max_steps=None, **kwargs):
    #     # Environments
    #     self.envs = [CrossingEnv(size=size, max_steps=max_steps, **kwargs),
    #                  DoorKeyEnv(size=size, max_steps=max_steps, **kwargs)]
        
    #     # Which environment is currently active (0 for CrossingEnv, 1 for DoorKeyEnv)
    #     self.current_env_idx = 0  # starting with CrossingEnv
        
    #     # Episode counter
    #     self.episode_count = 0
        
    #     # Number of episodes before switching
    #     self.t = t
        
    # def reset(self):
    #     # Check if it's time to switch environments
    #     if self.episode_count > 0 and self.episode_count % self.t == 0:
    #         self.current_env_idx = 1 - self.current_env_idx  # switch between 0 and 1
        
    #     self.episode_count += 1
        
    #     return self.envs[self.current_env_idx].reset()
    
    # def step(self, action):
    #     return self.envs[self.current_env_idx].step(action)
    
    # def render(self, *args, **kwargs):
    #     return self.envs[self.current_env_idx].render(*args, **kwargs)

    # You can add other methods as needed and forward them to the current environment.
    # ...


def blended_mission_helper():
    return blended_instance._gen_mission()

# Create an instance of the blended environment
# blended_env = BlendedEnv(t=5, size=8)
