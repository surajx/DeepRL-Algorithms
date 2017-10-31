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


def expert_policy_rollout(env_name, num_rollouts, expert_policy_file,
                          render=False, max_steps=None, save_filename=None):
    """Performs rollouts for the expert policy to generate data. Optionally the 
    generated data is saved. Rollout is a term used interchagenable with 
    trajectory, trial, or history.

    Args:
        env_name: Name of the environment (names recognizable by OpenAI gym)
        num_rollouts: Number of trials to be performed.
        render: Optional argument to display the rollouts, defaults to False.
        max_timestep: Optional argument to set the max timeteps for each rollout
        save_filename: Optional argument. If provided the generated data 
            would be saved to file.
    Returns:
        Dictionaary containing the generated data from the expert.
    """
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
    # TODO: Save expert data to file if enabled.
    return expert_data


def load_expert_data(expert_data_file):
    # TODO: Docstring
    pass


def get_args():
    # TODO: Docstring
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


def save_expert_data(expert_data, file_name='expert_data', ftype='pkl'):
    # TODO: docstring
    if ftype == 'pkl':
        pass  # TODO: Implement pickle saving of dict.
    elif ftype == 'csv':
        import csv
        with open(file_name + '.' + ftype, 'w', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(expert_data.keys())
            for i in range(len(expert_data['observations'])):
                csvwriter.writerow([expert_data[key][i]
                                    for key in expert_data.keys()])


def main():
    # TODO: docstring
    args = get_args()
    with tf.Session() as sess:
        tf_util.initialize()
        expert_data = expert_policy_rollout(args.envname, args.num_rollouts, args.expert_policy_file,
                                            render=args.render, max_steps=args.max_timesteps,
                                            save_filename=args.save_expert_data)


if __name__ == '__main__':
    main()
