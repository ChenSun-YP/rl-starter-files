from __future__ import annotations
from PIL import Image
import numpy as np
import datetime
from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs.crossing import CrossingEnv
from minigrid.envs.crossinggoodlava import CrossinggoodlavaEnv
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.envs.dynamicobstacles import DynamicObstaclesEnv
from minigrid.core.mission import MissionSpace
import logging
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
logger = logging.getLogger(__name__)
import torch
class BlendedEnv(MiniGridEnv):
    def __init__(
            self,
            t=100,
            size=3,
            max_steps=None,
            agent_view_size: int = 7,
            ground_truth_task=True, # the ground truth task is the task that the agent is trained on
            swap_seq = [1,2,3,4], # the sequence of the envs that the agent will be trained on
            **kwargs):
        # global blended_instance
        # blended_instance = self
        # Create a list of environments
        # if len(swap_seq) == 0:
        #     # generate a random swap sequence with lenghth of 20
        #     swap_seq = np.random.randint(0, 4, 20)
        print('swap_seq',swap_seq)
        print('init blended env with t', t)
        self.agent_view_size= 5
        self.swap_sqeuence = swap_seq
        self.swap_seq_idx = 0 
        self.envs = [
            DoorKeyEnv(size=7, max_steps=max_steps, **kwargs),
            CrossingEnv(size=7, max_steps=max_steps, **kwargs),
            DynamicObstaclesEnv(size=7, max_steps=max_steps, **kwargs),
            CrossinggoodlavaEnv(size=7, max_steps=max_steps, **kwargs)


        ]
        if ground_truth_task is True:
            #init the ground latent z for two subenvs with a 1*4 shape tensor
            # self.envs[0].ground_truth_z = np.array([[1, 0.25, 0.25, 0.25]])
            # self.envs[1].ground_truth_z = np.array([[0.25, 0.25, 0.25, 1]])
            # self.envs[2].ground_truth_z = np.array([[0.25, 1, 0.25, 0.25]])
            # self.envs[3].ground_truth_z = np.array([[0.25, 0.25, 1, 0.25]])
            self.envs[0].ground_truth_z = torch.tensor([1, 0.25, 0.25, 0.25])
            self.envs[1].ground_truth_z = torch.tensor([0.25, 0.25, 0.25, 1])
            self.envs[2].ground_truth_z = torch.tensor([0.25, 1, 0.25, 0.25])
            self.envs[3].ground_truth_z = torch.tensor([0.25, 0.25, 1, 0.25])



        # Which environment is currently active
        self.current_env_idx = 0
        self.current_env = self.envs[self.current_env_idx]
        self.episode_count = 0
        self.t = t
        mission_space = MissionSpace(mission_func=lambda: self.current_env._gen_mission())

        super().__init__(grid_size=size, mission_space=mission_space, **kwargs)
        self.step_count = 0  # Initialize step count for blended environment
        # image = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.agent_view_size, self.agent_view_size, 3),
        #     dtype="uint8",
        # )
        self.observation_space = spaces.Dict(
            {
                "image":  self.current_env.observation_space["image"],
                "direction":self.current_env.observation_space["direction"],
                "mission":  self.current_env.observation_space["mission"]
            }
        )
    def get_obs(self):
        return self.current_env.gen_obs() 

    def step(self, action):
        # Perform the step in the current environment
        self.current_obs = self.current_env.gen_obs() 
        self.step_count += 1  # Increment step count
        self.observation_space["image"] = self.current_env.observation_space["image"]
        self.observation_space["direction"] = self.current_env.observation_space["direction"]
        self.observation_space["mission"] = self.current_env.observation_space["mission"]
        obs, reward, done, truncated, info = self.current_env.step(action)
        if not self.observation_space.contains(obs):
            logger.error(f"Observation {obs} is not within the observation space.{self.observation_space}")

        # if done or truncated:
        #     # If the episode is done, reset the current environment
        #     obs = self.reset()
        #if the frame is a multiple of t, swap the environment  
        #fetch the current frame from the current env

        if self.step_count % self.t == 0:

            #render the current env save as a image
            # self.current_env.render_mode = 'rgb_array'
            # x= self.render()
            # self.render_and_save(self.current_env,save_path='before.png')
            next_env_idx = self.swap_sqeuence[self.swap_seq_idx] % len(self.envs)
            self.swap_seq_idx += 1
            prev_name =self.current_env.__class__.__name__
            obs = self.swap_env(obs,next_env_idx)
            # pring worker number
            print('swap happen in blend.py on step', self.step_count,'before env is',prev_name,'after this frame is',self.current_env.__class__.__name__)
            # trigger a log update outside to start color of the current env!


        return obs, reward, done, truncated, info
    def swap_env(self, obs,next_env_idx):
        # gen a unique swap id +current actual timestamp for logging purpose for everytime this is invoked with same seed
        swap_id = str(self.current_env_idx) + str(self.step_count) + str(datetime.datetime.now())

        # old = self.current_env.ground_truth_z
        # Get the current agent position from the observation's image
        before =self.get_env_name()
        agent_identifier = [1, 0, 0]
        agent_positions = np.argwhere(np.all(obs['image'] == agent_identifier, axis=-1))

        if agent_positions.shape[0] > 0:
            agent_pos = agent_positions[0]
        else:
            agent_pos = [0, 0]  # Default value in case the agent's position isn't found

        # Switch the environment now there are three envs
        #random a env index
        # self.current_env_idx = np.random.randint(0,len(self.envs))
        # self.current_env = self.envs[self.current_env_idx]
        self.current_env_idx = next_env_idx
        self.current_env = self.envs[self.current_env_idx]
        
        # Adjust agent_pos to be within the new environment's valid navigable area
        agent_pos[0] = min(max(agent_pos[0], 1), self.current_env.width - 2)
        agent_pos[1] = min(max(agent_pos[1], 1), self.current_env.height - 2)

        # Reset the new environment and place the agent in the same position
        new_obs = self.current_env.reset()
        # self.envs[previous_env_idx].reset()
        # self.envs[previous_env_idx].render()



    
        # rerender the new env
        #reset the previous env

        self.current_env.agent_pos = agent_pos
        self.current_env.agent_dir = obs['direction']  # Assuming the direction is the same
        # Regenerate the observation after changing the agent's position and direction
        new_obs = self.current_env.gen_obs()
        
        # self.render() 
        
        #print the old env latent z and updated one to check if it is updated
        # print(old,'switched to',self.current_env.ground_truth_z) 
        # print(f" swap {before ,self.get_env_name(),swap_id}")

        return new_obs
    # def swap_active_env(self):
    #     # Switch the environment
    #     self.current_env_idx = 1 - self.current_env_idx
    #     self.current_env = self.envs[self.current_env_idx]
    #     agent_identifier = [1, 0, 0]
    #     agent_positions = np.argwhere(np.all(self.current_obs['image'] == agent_identifier, axis=-1))
    #     # check if its a valid position
    #     if agent_positions.shape[0] > 0:
    #         agent_pos = agent_positions[0]
    #     else:
    #         agent_pos = [0, 0]
    #     # Adjust agent_pos to be within the new environment's valid navigable area
    #     agent_pos[0] = min(max(agent_pos[0], 1), self.current_env.width - 2)
    #     agent_pos[1] = min(max(agent_pos[1], 1), self.current_env.height - 2)
    #     # Reset the new environment and place the agent in the same position
    #     self.current_env.reset()
    #     self.current_env.agent_pos = agent_pos
    #     self.current_env.agent_dir = self.current_obs['direction']
        
        



    def reset(self, **kwargs):
        self.episode_count += 1
        # if self.episode_count % self.t == 0:
        #     self.swap_active_env()
        #     logger.info(f"reset Swapping environment to {self.current_env.__class__.__name__}")
        obs = self.current_env.reset(**kwargs)


        return obs


    

    # def render_and_save(self,current_env , save_path=None, ):
    #     rendered_image = current_env.render()
    #     print (rendered_image.shape)
    #     if rendered_image is None:
    #         print('rendered_image is None')
    #         return None
        
    #     # Save the rendered image as an image file
    #     if save_path:
    #         image = Image.fromarray(rendered_image)
    #         image.save(save_path)

    #     return rendered_image

    def render(self, *args, **kwargs):
        return self.current_env.render(*args, **kwargs)

    def _gen_mission(self):
        return self.current_env._gen_mission()
    def get_env_name(self):
        return self.current_env.__class__.__name__

    def get_ground_truth_latent_z(self):
        return self.current_env.ground_truth_z
    def get_ground_truth_env(self):
        return self.current_env_idx
    def get_ground_truth_env_label(self):
        return self.current_env.__class__.__name__
    def get_step(self):
        return self.step_count
    
        
