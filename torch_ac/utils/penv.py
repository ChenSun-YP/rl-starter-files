import multiprocessing
import gymnasium as gym
import torch
import copy

multiprocessing.set_start_method("fork")

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            if terminated or truncated:
                obs, _ = env.reset()
            conn.send((obs, reward, terminated, truncated, info))
        elif cmd == "reset":
            obs, _ = env.reset()
            conn.send(obs)
        elif cmd == "swap":# not used
            obs = env.swap_env(data)
            conn.send(obs)
        elif cmd == "get_ground_truth_latent_z":
            z = env.get_ground_truth_latent_z()
            conn.send(z)
        else:
            print("Invalid command: {}".format(cmd))
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = multiprocessing.Pipe()
            self.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()[0]] + [local.recv() for local in self.locals]
        return results

    def step(self, actions,latenszs=None):
        if latenszs is not None:
            for local, action in zip(self.locals, actions[1:]):
                local.send(("step", action))
            obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
            if terminated or truncated:
                obs, _ = self.envs[0].reset()
            results = zip(*[(obs, reward, terminated, truncated, info)] + [local.recv() for local in self.locals])
            return results
        else:
            for local, action in zip(self.locals, actions[1:]):
                local.send(("step", action))
            obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
            if terminated or truncated:
                obs, _ = self.envs[0].reset()
            results = zip(*[(obs, reward, terminated, truncated, info)] + [local.recv() for local in self.locals])
            return results


    def render(self):
        raise NotImplementedError


    def get_ground_truth_latent_z(self,num_proc):
        """
        Sends a command to get the ground truth latent z in a specific subprocess.

        Parameters:
        env_index : int
            The index of the environment to swap.
        """
        latent_zs = []
        # the first one is the main process
        latent_zs.append(self.envs[0].get_ground_truth_latent_z())
        # print('peny',self.envs[0].get_ground_truth_latent_z())

        for i in range(num_proc-1):
            self.locals[i].send(("get_ground_truth_latent_z", None))

            latent_zs.append(self.locals[i].recv())

        latent_zs = torch.stack(latent_zs)
        return latent_zs
    def get_ground_truth_env(self):
        """
        Sends a command to get the ground truth latent z in a specific subprocess.

        Parameters:
        env_index : int
            The index of the environment to swap.
        """
        # the first one is the main process
        return self.envs[0].current_env_idx

    def get_ground_truth_env_label(self):
        """
        Sends a command to get the ground truth latent z in a specific subprocess.

        Parameters:
        env_index : int
            The index of the environment to swap.
        """
        # the first one is the main process
        return self.envs[0].current_env.__class__.__name__



    # def get_fix_latent_z(self,num_proc):
    #     """
    #     Sends a command to get the ground truth latent z in a specific subprocess.

    #     Parameters:
    #     env_index : int
    #         The index of the environment to swap.
    #     """
    #     latent_zs = []
    #     for _ in range(num_proc):
    #         latent_zs.append(torch.tensor([1, 0.25, 0.25, 0.25])
    #     latent_zs = torch.stack(latent_zs)

    #     return latent_zs
    def get_main_env(self):

        # the first one is the main process
        return self.envs[0] , self.envs[0].current_env_idx
    def get_main_env_copy(self):
        # copy a main env and return it

        # the first one is the main process
        main_env_copy = copy.deepcopy(self.envs[0])

        return main_env_copy , main_env_copy.current_env_idx



    def swap_envs(self, env_index,current_obs):
        """
        Swaps the environment at the specified index by providing the current observation.

        Parameters:
        env_index : int
            The index of the environment to swap.
        """
        if env_index == 0:
            # Fetch the current observation and swap for the first environment
            new_obs = self.envs[0].swap_env(current_obs)
        else:
            # For other environments, get the current observation from the subprocess

            # Send swap command with the current observation
            self.locals[env_index - 1].send(("swap", current_obs))
            new_obs = self.locals[env_index - 1].recv()

        return new_obs