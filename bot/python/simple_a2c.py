# from build_order_env import BuildOrderEnv
from custom_env import CustomEnv
import os
import random
import math

import numpy as np
import gym.spaces.discrete
import gym
# import cppbot
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from custom_a2c import CustomA2C
from custom_env import CustomEnv

os.environ['KMP_DUPLICATE_LIB_OK']='True'

ACTIONS = 10
OUTPUT_DIM = 10 + 7


# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=[128, 128, dict(pi=[64], vf=[64])], feature_extraction="mlp")


step = 0
def simulate(model: A2C):
    global step
    step += 1
    if step % 100 != 0:
        return

    env = CustomEnv()
    # s = env.reset(True)
    s = env.reset(False)
    print("Start state:", s)
    done = False
    total_reward = 0
    values = []
    actions = []
    rewards = []
    while not done:
        action, _ = model.predict(s, deterministic=True)
        actions.append(action.item())
        values.append(0)
        s, reward, done, info = env.step(action)
        rewards.append(reward)
        total_reward += reward

    print(" ".join([f"{a}".rjust(2) for (a,v) in zip(actions, values)]))
    print(" ".join([f"{round(v,1)}".rjust(2) for (a,v) in zip(actions, values)]))
    print(" ".join([f"{reward}".rjust(2) for reward in rewards]))
    print(env.render())
    print(f"Optimal: {total_reward}")


def breakout_a2c():
    # try_optimal()

    print("Build environements")
    env = DummyVecEnv([(lambda: CustomEnv()) for i in range(16)])
    print("Done")
    # model = A2C(CustomPolicy, env, ent_coef=0.2, n_steps=5, gamma=0.999, verbose=1, learning_rate=0.0005, tensorboard_log="./a2c_test_tensorboard/")
    model = PPO2(CustomPolicy, env, ent_coef=0.01, n_steps=50, gamma=0.999, verbose=1, learning_rate=0.0005, tensorboard_log="./a2c_test_tensorboard/")
    model.learn(total_timesteps=10000000, callback=lambda a,b: simulate(model))


def main():
    breakout_a2c()

if __name__ == "__main__":
    main()
