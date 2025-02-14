from abc import ABC, abstractmethod
import torch
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        self.prev_obss = [None for _ in range(self.num_frames_per_proc)]
        self.prev_obss[0] = self.obs

        # self.prev_obss[0] = self.preprocess_obss(self.obs, device=self.device)

        for i in range(self.num_frames_per_proc):
            # get ground truth latent z for current env
            # find latent_z = self.env[i].current_env.latent_z where env is ParallelEnv(envs)

            # where do i get the latent_z for every proc in current frame?? od i get i t all at once?

            # Do one agent-environment interaction
            # Store the current observation before taking the action for the next step
            if i < self.num_frames_per_proc - 1:
                self.prev_obss[i + 1] = self.obs

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))


            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)
            
            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
        if self.prev_obss[0] is None:
            self.prev_obss[0] = self.preprocess_obss(self.obs, device=self.device)

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.prev_obs = [self.prev_obss[i][j]
                            for j in range(self.num_procs)
                            for i in range(self.num_frames_per_proc)]

        # exps.prev_obs = []
        # for obs in self.prev_obss:
        #     if obs is not None:
        #         exps.prev_obs.append(self.preprocess_obss(obs, device=self.device))
        #     else:
        #         # Handle the case where the first observation is None
        #         exps.prev_obs.append(self.preprocess_obss(self.obs, device=self.device))


        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences
  
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        # Preprocess prev_obs if not already preprocessed
        exps.prev_obs = self.preprocess_obss(exps.prev_obs, device=self.device)
        


        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
    
    def collect_experiences_latent(self,latent_z=None):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        self.prev_obss = [None for _ in range(self.num_frames_per_proc)]
        self.prev_obss[0] = self.obs

        # self.prev_obss[0] = self.preprocess_obss(self.obs, device=self.device)
        def gen_fake_latent(num_procs=16):
                        latent_zs = []
                        for _ in range(num_procs):
                            latent_zs.append(torch.tensor([1, 0.25, 0.25, 0.25])) #hard coded
                        latent_z = torch.stack(latent_zs)
                        return latent_z
        for i in range(self.num_frames_per_proc):
            # get ground truth latent z for current env
            # find latent_z = self.env[i].current_env.latent_z where env is ParallelEnv(envs)

            # where do i get the latent_z for every proc in current frame?? od i get i t all at once?
            if latent_z is not None:
                # give fixed latent
                # latent_zs = []
                # for _ in range(self.num_procs):
                #     latent_zs.append(latent_z)
                # batch_latent_z = torch.stack(latent_zs)
                batch_latent_z = gen_fake_latent(self.num_procs)
            else:
                batch_latent_z = self.env.get_ground_truth_latent_z(self.num_procs)



            # Do one agent-environment interaction
            # Store the current observation before taking the action for the next step
            if i < self.num_frames_per_proc - 1:
                self.prev_obss[i + 1] = self.obs

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    # print('base',batch_latent_z)
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),latent_z=batch_latent_z)
                else:
                    dist, value = self.acmodel(preprocessed_obs,latent_z=latent_z)
            action = dist.sample()
            
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))
            # if total frame is 3000, do a envswap
            # if self.num_frames == 30:
            #     #swap all envs  at the same time
            #     print('swapping')
            #     obs = self.env.swap_env(obs)
            #     print('swapping env' ,self.env.current_env.__class__.__name__)
            #     # Update experiences values
            #     self.obss[i] = self.obs

            # for proc_id in range(self.num_procs):
            #     if self.should_swap(proc_id):  # Implement this method to determine when to swap
            #         self.env.swap_env(proc_id)
            # for i in range(len(self.env.locals) + 1):
            #     if 1==1:  # Replace with your specific condition
            #         # Assuming each local environment is a BlendedEnv instance
            #         # Trigger the swap
            #         print('Attempting to swap environment:', i)
            #         self.env.swap_envs(i,obs[i])

                # print('swapping from', self.env.current_env.__class__.__name__)
                # obs = self.env[i].swap_env(obs)
                # print('swapping env' ,self.env[i].current_env.__class__.__name__)
            # Update experiences values 
            

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)
            
            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
        if self.prev_obss[0] is None:
            self.prev_obss[0] = self.preprocess_obss(self.obs, device=self.device)

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),latent_z = batch_latent_z)
            else:
                _, next_value = self.acmodel(preprocessed_obs,latent_z = batch_latent_z)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.prev_obs = [self.prev_obss[i][j]
                            for j in range(self.num_procs)
                            for i in range(self.num_frames_per_proc)]

        # exps.prev_obs = []
        # for obs in self.prev_obss:
        #     if obs is not None:
        #         exps.prev_obs.append(self.preprocess_obss(obs, device=self.device))
        #     else:
        #         # Handle the case where the first observation is None
        #         exps.prev_obs.append(self.preprocess_obss(self.obs, device=self.device))


        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences
  
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        # Preprocess prev_obs if not already preprocessed
        exps.prev_obs = self.preprocess_obss(exps.prev_obs, device=self.device)
        


        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
    
    import imageio
    import matplotlib.pyplot as plt
    import numpy as np

    def save_gif(self,update_name,model_dir, steps=100):
        # Initialize the environment and get the initial state
        frames = []
        env = self.env.get_main_env()
        env.reset()
        for _ in range(steps):
            # Render the environment to a numpy array
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            # Perform a  action base on acmodel, returns the new state, reward and whether the game is over
            action = self.acmodel.get_action(obs)
            state, action, reward, done, info = env.step(action)

            # Use collect_experiences to interact with the environment

            if done:
                return 2
                break

        # Save frames as a GIF at right directory with right name
        return 1

        # Save frames as a GIF at right directory with right name
        imageio.mimsave(model_dir + update_name + '.gif', frames, duration=1/30)
            
    def get_env(self):
        env , env_idx=  self.env.get_main_env()
        return env , env_idx
        

    # In your main thread, you can call this function with your model and environment
    # save_gif(my_model, my_env)
    @abstractmethod
    def update_parameters(self):
        pass
