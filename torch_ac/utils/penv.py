import multiprocessing
import gymnasium as gym
import torch

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


    # def swap_env(self, env_index, new_env):
    #     """
    #     not used
    #     Swaps out the environment at the specified index with a new environment.

    #     Parameters:
    #     env_index : int
    #         The index of the environment to swap.
    #     new_env : gym.Env
    #         The new environment to use.
    #     """
    #     # Check if it's the first environment, which is not in a separate process
    #     if env_index == 0:
    #         self.envs[0].close()  # Close the current environment
    #         self.envs[0] = new_env  # Assign the new environment
    #     else:
    #         # For other environments, which are in separate processes
    #         self.locals[env_index - 1].send(("close", None))  # Send close command to the process
    #         self.locals[env_index - 1].close()  # Close the local connection

    #         # Start a new process with the new environment
    #         local, remote = multiprocessing.Pipe()
    #         self.locals[env_index - 1] = local
    #         p = multiprocessing.Process(target=worker, args=(remote, new_env))
    #         p.daemon = True
    #         p.start()
    #         remote.close()
    # def swap_envs(self, env_index):
    #     """
    #     Sends a command to swap the environment in a specific subprocess.

    #     Parameters:
    #     env_index : int
    #         The index of the environment to swap.
    #     """
    #     if env_index == 0:
    #         #get obs
    #         obs = self.envs[0].reset()
    #         # Directly swap the environment in the main process
    #         self.envs[0].swap_env(obs)
    #     else:
    #         #get obs
    #         obs = self.envs[0].reset()
    #         # Send swap command to the subprocess
    #         self.locals[env_index - 1].send(("swap", None))
    #         # Receive confirmation (e.g., the new observation after swapping)
    #         new_obs = self.locals[env_index - 1].recv()
    #         return new_obs

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
        for i in range(num_proc-1):
            self.locals[i].send(("get_ground_truth_latent_z", None))
            latent_zs.append(self.locals[i].recv())
        latent_zs = torch.stack(latent_zs)


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
        return self.envs[0]




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