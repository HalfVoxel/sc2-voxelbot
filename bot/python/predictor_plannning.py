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
from stable_baselines.common.policies import MlpPolicy
from custom_a2c import CustomA2C
from custom_env import CustomEnv
import simulator_visualizer
import tensorflow as tf
import tensorflow.keras.layers as keras
from stable_baselines.a2c.utils import conv, linear
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

            minimap = self.processed_obs[:,:MINIMAP_SIZE*MINIMAP_SIZE*NUM_MINIMAPS]
            minimap = tf.reshape(minimap, [-1, NUM_MINIMAPS, MINIMAP_SIZE, MINIMAP_SIZE], name="reshape_minimap")
            meta = self.processed_obs[:,16*16*4:]

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
        env = DummyVecEnv([(lambda: RLPlanningEnv()) for i in range(16)])
        print("Done")
        # model = A2C(CustomPolicy2, env, ent_coef=0.01, n_steps=20, gamma=0.99, verbose=1, learning_rate=0.0005, vf_coef=0.001, tensorboard_log="./tensorboard/planning")
        model = PPO2(CustomPolicy2, env, ent_coef=0.01, n_steps=20, gamma=0.99, verbose=1, vf_coef=0.01, learning_rate=0.001, tensorboard_log="./tensorboard/planning")
        model.learn(total_timesteps=10000000, callback=lambda a,b: simulate(model), tb_log_name=f"PPO2 {comment}")

    def visualize(epoch):
        pass

    common.train_interface(cache, train, visualize)

def main():
    breakout_a2c()

if __name__ == "__main__":
    main()
