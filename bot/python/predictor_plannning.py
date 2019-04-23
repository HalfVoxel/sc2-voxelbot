# from build_order_env import BuildOrderEnv
from rl_planning_env import RLPlanningEnv, MINIMAP_SIZE, NUM_MINIMAPS
import os
import random
import math
import sys, select
import numpy as np
import gym.spaces.discrete
import gym
from stable_baselines.common.policies import FeedForwardPolicy, ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import DQNPolicy
from custom_a2c import CustomA2C
from custom_env import CustomEnv
import simulator_visualizer
import tensorflow as tf
import tensorflow.keras.layers as keras
from stable_baselines.a2c.utils import conv, linear
import stable_baselines
import tensorflow.contrib.layers as tf_layers

import common

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["MPLBACKEND"] = "TkAgg"


# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=[128, 128, dict(pi=[64], vf=[64])], feature_extraction="mlp")


def small_cnn(scaled_images, **kwargs):
    """
    Smaller than:
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu

    with tf.variable_scope('cnn'):
        layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=3, stride=2, init_scale=np.sqrt(2), data_format="NCHW"), name="relu")
        layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=4, stride=2, init_scale=np.sqrt(2), data_format="NCHW"), name="relu")
        # layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), data_format="NCHW"))
        layer_3 = keras.Flatten()(layer_2)
        return activ(linear(layer_3, 'fc1', n_hidden=16, init_scale=np.sqrt(2)), name="relu")

class CustomPolicy2(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy2, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False)

        # Model
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            unit_offset = MINIMAP_SIZE*MINIMAP_SIZE*NUM_MINIMAPS
            minimap = self.processed_obs[:,:unit_offset]
            # units = self.processed_obs[unit_offset:unit_offset + 3*100]
            meta = self.processed_obs[:,16*16*4:]

            minimap = tf.reshape(minimap, [-1, NUM_MINIMAPS, MINIMAP_SIZE, MINIMAP_SIZE], name="reshape_minimap")
            # units = tf.reshape(units, [-1, 100, 3], name="reshape_units")
            # unit_types = units[:,:,2]
            # coords = (units[:,:,0]*MINIMAP_SIZE + units[:,:,1]).to_int32()
            # unit_embeddings = keras.Embedding(35, 8, name="unit_embedding")(unit_types.to_int32())
            # embedding_minimap = tf.zeros([-1, MINIMAP_SIZE*MINIMAP_SIZE], dtype=tf.float32)
            # embedding_minimap = tf.tensor_scatter_add(embedding_minimap, coords, unit_embedding)

            extracted_features = small_cnn(minimap * 0.1)
            extracted_features = keras.Flatten()(extracted_features)
            extracted_features = keras.concatenate([extracted_features, meta], axis=1)

            extracted_features = activ(tf.layers.dense(extracted_features, 64, name='fc1'), name="relu")
            extracted_features = activ(tf.layers.dense(extracted_features, 64, name='fc2'), name="relu")

            with tf.variable_scope('policy_head'):
                pi_h = extracted_features
                for i, layer_size in enumerate([64]):
                    pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)), name="relu")
                pi_latent = pi_h
            
            with tf.variable_scope('value_head'):
                vf_h = extracted_features
                for i, layer_size in enumerate([64]):
                    vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)), name="relu")
                value_fn = tf.layers.dense(vf_h, 1, name='vf')
                vf_latent = vf_h

            self.proba_distribution, self.policy, self.q_value = self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

        param_count = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print(f"Parameter count: {param_count}")

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})

class FeedForwardPolicy2(DQNPolicy):
    """
    Policy object that implements a DQN policy, using a feed forward neural network.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param layer_norm: (bool) enable layer normalisation
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp",
                 obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu, **kwargs):
        super(FeedForwardPolicy2, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse,
                                                scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            layers = [64, 64]
        
                # Model
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            unit_offset = MINIMAP_SIZE*MINIMAP_SIZE*NUM_MINIMAPS
            minimap = self.processed_obs[:,:unit_offset]
            # units = self.processed_obs[unit_offset:unit_offset + 3*100]
            meta = self.processed_obs[:,16*16*4:]

            minimap = tf.reshape(minimap, [-1, NUM_MINIMAPS, MINIMAP_SIZE, MINIMAP_SIZE], name="reshape_minimap")
            # units = tf.reshape(units, [-1, 100, 3], name="reshape_units")
            # unit_types = units[:,:,2]
            # coords = (units[:,:,0]*MINIMAP_SIZE + units[:,:,1]).to_int32()
            # unit_embeddings = keras.Embedding(35, 8, name="unit_embedding")(unit_types.to_int32())
            # embedding_minimap = tf.zeros([-1, MINIMAP_SIZE*MINIMAP_SIZE], dtype=tf.float32)
            # embedding_minimap = tf.tensor_scatter_add(embedding_minimap, coords, unit_embedding)

            extracted_features = small_cnn(minimap * 0.1)
            extracted_features = keras.Flatten()(extracted_features)
            extracted_features = keras.concatenate([extracted_features, meta], axis=1)

            extracted_features = activ(tf.layers.dense(extracted_features, 64, name='fc1'), name="relu")
            extracted_features = activ(tf.layers.dense(extracted_features, 64, name='fc2'), name="relu")

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    action_out = extracted_features
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    action_out = extracted_features
                    for layer_size in layers:
                        action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                        action_out = act_fun(action_out)

                action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores

        self.q_values = q_out
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

class CustomDQNPolicy2(DQNPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomDQNPolicy2, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False, dueling=True, obs_phs=None)

        # Model
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            unit_offset = MINIMAP_SIZE*MINIMAP_SIZE*NUM_MINIMAPS
            minimap = self.processed_obs[:,:unit_offset]
            # units = self.processed_obs[unit_offset:unit_offset + 3*100]
            meta = self.processed_obs[:,16*16*4:]

            minimap = tf.reshape(minimap, [-1, NUM_MINIMAPS, MINIMAP_SIZE, MINIMAP_SIZE], name="reshape_minimap")
            # units = tf.reshape(units, [-1, 100, 3], name="reshape_units")
            # unit_types = units[:,:,2]
            # coords = (units[:,:,0]*MINIMAP_SIZE + units[:,:,1]).to_int32()
            # unit_embeddings = keras.Embedding(35, 8, name="unit_embedding")(unit_types.to_int32())
            # embedding_minimap = tf.zeros([-1, MINIMAP_SIZE*MINIMAP_SIZE], dtype=tf.float32)
            # embedding_minimap = tf.tensor_scatter_add(embedding_minimap, coords, unit_embedding)

            extracted_features = small_cnn(minimap * 0.1)
            extracted_features = keras.Flatten()(extracted_features)
            extracted_features = keras.concatenate([extracted_features, meta], axis=1)

            extracted_features = activ(tf.layers.dense(extracted_features, 64, name='fc1'), name="relu")
            extracted_features = activ(tf.layers.dense(extracted_features, 64, name='fc2'), name="relu")

            with tf.variable_scope('policy_head'):
                pi_h = extracted_features
                for i, layer_size in enumerate([64]):
                    pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
                
                pi_h = tf.layers.dense(pi_h, self.n_actions, name='vf')
                pi_latent = pi_h
            
            with tf.variable_scope('value_head'):
                vf_h = extracted_features
                for i, layer_size in enumerate([64]):
                    vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)), name="relu")
                value_fn = tf.layers.dense(vf_h, 1, name='vf')
            
            action_scores_mean = tf.reduce_mean(pi_latent, axis=1)
            action_scores_centered = pi_latent - tf.expand_dims(action_scores_mean, axis=1)
            q_out = value_fn + action_scores_centered

            # self.proba_distribution, self.policy, self.q_values = self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
            self.q_values = q_out

        # self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

        param_count = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print(f"Parameter count: {param_count}")

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


step = 0
def simulate(model: A2C):
    global step
    step += 1
    # if step % 100 != 0:
    #     return
    
    i, o, e = select.select( [sys.stdin], [], [], 0 )
    if not i or sys.stdin.readline().strip() != "d":
        return
    
            
    env = RLPlanningEnv()
    # s = env.reset(True)
    s = env.reset(False)
    print("Start state:", s)
    done = False
    total_reward = 0
    values = []
    actions = []
    rewards = []
    simulator_visualizer.reset_visualizer()
    while not done:
        # env.render()
        action, _ = model.predict(s, deterministic=random.random() > 0.9)
        actions.append(action.item())
        values.append(0)
        s, reward, done, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        print(f"Action: {env.action_name(action)} Reward: {reward}")

        units, state_time = env.visualization_info()
        simulator_visualizer.visualize(units, state_time, 0, 0, s)

        i, o, e = select.select( [sys.stdin], [], [], 0 )
        if i and sys.stdin.readline().strip() == "c":
            break

    print(" ".join([f"{a}".rjust(2) for (a,v) in zip(actions, values)]))
    print(" ".join([f"{round(v,1)}".rjust(2) for (a,v) in zip(actions, values)]))
    print(" ".join([f"{reward}".rjust(2) for reward in rewards]))
    # print(env.render())
    print(f"Optimal: {total_reward}")


def breakout_a2c():
    # try_optimal()
    def cache():
        pass

    def train(comment):
        print("Build environements")
        # env = DummyVecEnv([(lambda: RLPlanningEnv()) for i in range(16)])
        print("Done")
        # model = A2C(CustomPolicy2, env, ent_coef=0.01, n_steps=20, gamma=0.99, verbose=1, learning_rate=0.0005, vf_coef=0.001, tensorboard_log="./tensorboard/planning")
        # model = PPO2(CustomPolicy2, env, ent_coef=0.01, n_steps=20, gamma=0.99, verbose=1, vf_coef=0.2, learning_rate=0.001, tensorboard_log="./tensorboard/planning")
        env = DummyVecEnv([(lambda: RLPlanningEnv()) for i in range(1)])
        # model = DQN(stable_baselines.deepq.policies.MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard/planning")
        model = DQN(FeedForwardPolicy2, env, verbose=1, tensorboard_log="./tensorboard/planning")
        model.learn(total_timesteps=1000000, callback=lambda a,b: simulate(model), tb_log_name=f"DQN {comment}")

    def visualize(epoch):
        pass

    common.train_interface(cache, train, visualize)

def main():
    breakout_a2c()

if __name__ == "__main__":
    main()
