"""
Code to run behavioral cloning on genererated expert data.
Example usage:
    Create Fresh Data:
    python behaviour_cloning.py Hopper-v1 \
    --expert_policy_file experts/Hopper-v1.pkl \
    --save_expert_data gen_data/expert_data_hopper.pkl \
    --render --num_rollouts 80

    Load Saved Data:
    python behaviour_cloning.py Hopper-v1 \
        --load_expert_data gen_data/expert_data_hopper.pkl \
        --render --num_rollouts 80

Implementation of Behaviour Cloning: Suraj Narayanan Sasikumar
"""

import util
import math
import numpy as np
import expert_policy
import tensorflow as tf


class BCPolicy(object):
    """Create a policy using behviour cloning from expert data."""

    def __init__(self, oa_dim, max_steps=2000):
        # TODO: docstring

        # Attributes
        self.o_dim, self.a_dim = oa_dim
        self.max_steps = max_steps

        # Constants
        self.HIDDEN_ARCH = [128, 32]
        self.beta = 0.01  # Regularization coefficient

        # Build Policy Graph
        self._gen_full_graph()

        # Instantiate Default session
        self.sess = tf.Session(graph=self.bc_graph)

        # Intialize all the variables
        self.sess.run(self.init)

    def _gen_full_graph(self):
        # TODO: docstring
        self.bc_graph = tf.Graph()
        with self.bc_graph.as_default():
            # Placeholders for obs and act
            self.obs_ph = tf.placeholder(tf.float32, shape=(1, self.o_dim))
            self.act_ph = tf.placeholder(tf.float32, shape=(1, self.a_dim))

            # Inference graph
            self.logit = self._inference_graph(self.obs_ph)

            # Loss graph
            self.loss = self._loss_graph(self.logit, self.act_ph)

            # Train graph
            self.train_op = self._train_graph(self.loss)

            # Init all variables.
            self.init = tf.global_variables_initializer()

    def _get_W_and_b(self, shape, regularizer=None):
        # TODO: docstring
        # Samples from truncated normal are bounded at two stddev to either side
        # of the mean.
        # Initialization Justifications:
        # 1. Random initialization helps break symmetry between learned features.
        # 2. Bounded values(truncated) help to control the magnitude of the
        #    gradients, resulting in better convergence.
        # 3. ReLU adjusted Xavier Initialization [1][2][3][4]
        #    var[W_i] = \sqrt{2/(number of inputs to neuron)}
        # [1] https://arxiv.org/pdf/1502.01852.pdf
        # [2] http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # [3] http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        # [4] http://deepdish.io/2015/02/24/network-initialization
        weights = tf.get_variable("weights", initializer=tf.truncated_normal(
            shape, stddev=math.sqrt(2.0 / float(shape[0]))),
            regularizer=regularizer)
        bias = tf.Variable(tf.zeros([shape[1]]), name="bias")
        return weights, bias

    def _inference_graph(self, obs):
        # TODO: docstring
        # TODO: can't we make this into a loop over HIDDEN_ARCH?
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
        with tf.variable_scope('hidden_layer_1'):
            weights, bias = self._get_W_and_b(
                [self.o_dim, self.HIDDEN_ARCH[0]], regularizer)
            hidden_1_op = tf.nn.relu(tf.matmul(obs, weights) + bias)
        with tf.variable_scope('hiddel_layer_2'):
            weights, bias = self._get_W_and_b(
                [self.HIDDEN_ARCH[0], self.HIDDEN_ARCH[1]], regularizer)
            hidden_2_op = tf.nn.relu(tf.matmul(hidden_1_op, weights) + bias)
        with tf.variable_scope('logit_layer'):
            weights, bias = self._get_W_and_b(
                [self.HIDDEN_ARCH[1], self.a_dim], regularizer)
            logit = tf.matmul(hidden_2_op, weights) + bias
        return logit

    def _loss_graph(self, logit, act):
        # TODO: docstring
        loss = tf.losses.mean_squared_error(act, logit)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_variables)
        return loss + reg_loss

    def _train_graph(self, loss):
        # TODO: docstring
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train = optimizer.minimize(loss, global_step=global_step)
        return train

    def _get_data(self, expert_data):
        # TODO: docstring
        # Preprocessing
        def _resize_obs(obs, act):
            obs_rs = tf.reshape(obs, (1, -1))
            return obs_rs, act

        # Assume that each row of `observations` corresponds to the
        # same row as `actions`.
        assert expert_data['observations'].shape[0] == expert_data['actions'].shape[0]
        with self.bc_graph.as_default():
            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (expert_data['observations'], expert_data['actions']))
            dataset = dataset.map(_resize_obs)
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            nxt_obs, nxt_act = iterator.get_next()
        return nxt_obs, nxt_act

    def train(self, expert_data):
        # TODO: docstring
        nxt_obs, nxt_act = self._get_data(expert_data)
        for step in range(self.max_steps):
            obs = self.sess.run(nxt_obs)
            act = self.sess.run(nxt_act)
            feed_dict = {
                self.obs_ph: obs,
                self.act_ph: act
            }
            _, loss_value = self.sess.run(
                [self.train_op, self.loss], feed_dict=feed_dict)
            if step % 100 == 0:
                print('Step %d: loss = %.2f' % (step, loss_value))

    def sample(self, obs):
        # TODO: docstring
        feed_dict = {
            self.obs_ph: obs
        }
        return self.sess.run(self.logit, feed_dict=feed_dict)


def get_args():
    """Parses Command line arguments

    Returns:
        Namespace object containing arguments and their values.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)  # TODO: help
    # parser.add_argument('--bc-hidden-layers', nargs='+', type=int,
    #                     help="Hidden layer structure Ex: --hidden-layers 128 64 32")
    # parser.add_argument('--bc-learning-rate', type=float)  # TODO: help
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


def rollout_policy(policy, env, args):
    max_steps = env.spec.timestep_limit
    returns, observations, actions = [], [], []

    # Start rollouts
    for i in range(args.num_rollouts):
        print('Iteration:', i)
        obs = env.reset()  # First observation for the current rollout
        done = False  # bool that tracks the end of rollout.
        cum_reward = 0.
        steps = 0
        # Interact with env until rollout is terminated.
        while not done:
            action = policy.sample(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, rew, done, _ = env.step(action)  # _ is diagnostic information
            cum_reward += rew
            steps += 1
            if args.render:
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


def main():
    args = get_args()  # Get commandline arguments.
    expert_data = expert_policy.get_expert_data(args)  # Generate expert data.
    env, oa_dim = util.get_env(args.env_name)
    policy = BCPolicy(oa_dim, max_steps=50000)  # Untrained policy pi(a|o)
    policy.train(expert_data)  # Train policy using Behaviour Cloning
    rollout_policy(policy, env, args)


if __name__ == '__main__':
    main()
