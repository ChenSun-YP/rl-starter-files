import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None, **kwargs):
    env = gym.make(env_key, render_mode=render_mode, **kwargs)
    env.reset(seed=seed)
    return env
