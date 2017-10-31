#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
Restructured for readability and impl of Behaviour Cloning: Suraj Narayanan Sasikumar
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


def get_expert_policy(expert_policy_file):
    """Loads the saved expert policy

    Args:
        expert_policy_file (str): The file name of the saved expert policy.

    Returns:
        The policy function.
    """
    print('Building expert policy...', expert_policy_file)
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('Built and Loaded.')
    return policy_fn


def init_gym(env_name):
    """Initialize environment from gym.

    Args:
        env_name (str): Name of the environment to be initialized.

    Returns:
        The gym environement object.
    """
    import gym
    env = gym.make(env_name)
    return env


def expert_policy_rollout(args):
    """Performs rollouts for the expert policy to generate data. Optionally the
    generated data is saved. Rollout is a term used interchagenable with
    trajectory, trial, or history.

    Args:
        args: Configuration values from the command line.
    Returns:
        Dictionary containing the generated data from the expert.
    """
    # Init variables from args
    env_name = args.envname  # Name of the environment.
    num_rollouts = args.num_rollouts  # Number of trials to be performed.
    expert_policy_file = args.expert_policy_file  # Saved expert policy file.
    render = args.render  # Display agent, defaults to False.
    max_steps = args.max_timesteps  # Max timeteps for each rollout
    save_filename = args.save_expert_data  # Filename to store expert data.

    # Setup gym environment
    env = init_gym(env_name)
    max_steps = max_steps or env.spec.timestep_limit
    returns, observations, actions = [], [], []

    # Setup expert policy
    policy_fn = get_expert_policy(expert_policy_file)

    # Start rollouts
    for i in range(num_rollouts):
        print('Iteration:', i)
        obs = env.reset()  # Gives the first observation for the current rollout
        done = False  # bool that tracks the end of rollout.
        cum_reward = 0.
        steps = 0
        # Interact with env until rollout is terminated.
        while not done:
            action = policy_fn(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, rew, done, _ = env.step(action)  # _ is diagnostic information
            cum_reward += rew
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(cum_reward)
    # Return analysis
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}
    if save_filename:
        save_expert_data(expert_data, file_name=save_filename)
    return expert_data


def load_expert_data(expert_data_file):
    """Loads the exper_data from file"""
    with open(expert_data_file, 'rb') as f:
        expert_data = pickle.loads(f.read())
    return expert_data


def get_args():
    """Parses Command line arguments

    Args:
        None

    Returns:
        Namespace object containing arguments and their values.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)  # TODO: help
    parser.add_argument('--expert_policy_file', type=str,
                        help='Expert Policy to load and perform rollout.')
    parser.add_argument('--save_expert_data', type=str,
                        help='File name to save generated expert data.')
    parser.add_argument('--load_expert_data', type=str,
                        help='File name for saved expert policy rollout.')
    parser.add_argument('--render', action='store_true')  # TODO: help
    parser.add_argument('--max_timesteps', type=int)  # TODO: help
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    return args


def save_expert_data(expert_data, file_name='expert_data.pkl'):
    """Save generated expert data to file.

    Args:
        expert_data (dict): generated expert data.

    Returns:
        None
    """
    with open(file_name, 'wb') as f:
        pickle.dump(expert_data, f)


def run_expert(args):
    # TODO: docstring
    if args.load_expert_data:
        expert_data = load_expert_data(args.load_expert_data)
    else:
        expert_data = expert_policy_rollout(args)
    return expert_data


def gen_input_graph(expert_data):
    # TODO: docstring
    # Assume that each row of `observations` corresponds to the same row as `actions`.
    # Preprocessing
    def _resize_obs(obs, act):
        obs_rs = tf.reshape(obs, (1, -1))
        return obs_rs, act
    assert expert_data['observations'].shape[0] == expert_data['actions'].shape[0]
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        (expert_data['observations'], expert_data['actions']))
    dataset = dataset.map(_resize_obs)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def gen_inference_graph(data_tf):
    # TODO: docstring
    # Generate inference graph


def main():
    """ Entry point for the program.
    """
    args = get_args()
    # Build inference graph
    # Build training graph
    with tf.Session() as sess:
        tf_util.initialize()
        expert_data = run_expert(args)
        next_data = gen_input_graph(expert_data)
        for i in range(10):
            print(sess.run(next_data)[0].shape)
            print(sess.run(next_data)[1].shape)


if __name__ == '__main__':
    main()
