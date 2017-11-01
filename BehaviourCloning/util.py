"""
Utility file for commonly used functions.

Author: Suraj Narayanan Sasikumar
"""


def get_env(env_name):
    """Initialize environment from gym.

    Args:
        env_name (str): Name of the environment to be initialized.

    Returns:
        The gym environment object along with the dimensions of the 
        observation and action space.
    """
    import gym
    env = gym.make(env_name)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    return env, [o_dim, a_dim]
