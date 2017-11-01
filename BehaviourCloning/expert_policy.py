"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python expert_policy.py  Hopper-v1 \
        --expert_policy_file experts/Hopper-v1.pkl --num_rollouts 20 \
        --save_expert_data gen_data/expert_data_hopper.pkl

Authors:
    This script and included expert policies: Jonathan Ho (hoj@openai.com)
    Restructured for readability and extension: Suraj Narayanan Sasikumar
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import load_policy
import util


class ExpertPolicy(object):
    """ Build, Load, and generate data using provided expert policy"""

    def __init__(self, gym_env, args):
        self.env = gym_env
        # Number of trials to be performed.
        self.num_rollouts = args.num_rollouts
        # Saved expert policy file.
        self.expert_policy_file = args.expert_policy_file
        self.render = args.render  # Display agent, defaults to False.
        self.max_steps = args.max_timesteps  # Max timeteps for each rollout
        # Filename to store expert data.
        self.save_filename = args.save_expert_data

    def run_expert(self):
        # TODO: docstring
        return self._expert_policy_rollout()

    def get_expert_policy(self):
        """Loads the saved expert policy

        Returns:
            The policy function.
        """
        print('Building expert policy....')
        policy_fn = load_policy.load_policy(self.expert_policy_file)
        return policy_fn

    def _expert_policy_rollout(self):
        """Performs rollouts for the expert policy to generate data. 
        Optionally the generated data is saved. Rollout is a term used 
        interchagenable with trajectory, trial, or history.

        Returns:
            Dictionary containing the generated data from the expert.
        """
        max_steps = self.max_steps or self.env.spec.timestep_limit
        returns, observations, actions = [], [], []

        # Setup expert policy
        policy_fn = self.get_expert_policy()

        # Start rollouts
        for i in range(self.num_rollouts):
            print('Iteration:', i)
            obs = self.env.reset()  # First observation for the current rollout
            done = False  # bool that tracks the end of rollout.
            cum_reward = 0.
            steps = 0
            # Interact with env until rollout is terminated.
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, rew, done, _ = self.env.step(
                    action)  # _ is diagnostic information
                cum_reward += rew
                steps += 1
                if self.render:
                    self.env.render()
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
        if self.save_filename:
            self.save_expert_data(expert_data, file_name=self.save_filename)
        return expert_data

    def save_expert_data(self, expert_data, file_name='gen_data/expert_data.pkl'):
        """Save generated expert data to file.

        Args:
            expert_data (dict): generated expert data.
        """
        with open(file_name, 'wb') as f:
            pickle.dump(expert_data, f)


def load_expert_data(expert_data_file):
    """Loads the expert data from file

    Args:
        expert_data_file (str): Saved expert data pickle filename.

    Returns:
        Expert data dictionary.
    """
    with open(expert_data_file, 'rb') as f:
        expert_data = pickle.loads(f.read())
    return expert_data


def get_expert_data(args):
    # TODO: docstring
    if args.load_expert_data:
        expert_data = load_expert_data(args.load_expert_data)
    else:
        env, _ = util.get_env(args.env_name)
        with tf.Session():
            tf_util.initialize()
            expert_data = ExpertPolicy(env, args).run_expert()
    return expert_data


def _get_args():
    """Parses Command line arguments

    Returns:
        Namespace object containing arguments and their values.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)  # TODO: help
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


if __name__ == '__main__':
    get_expert_data(_get_args())
