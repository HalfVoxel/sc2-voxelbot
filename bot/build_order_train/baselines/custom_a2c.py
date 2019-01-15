import time
from stable_baselines.a2c.utils import Scheduler, total_episode_reward_logger
from stable_baselines.common import explained_variance, SetVerbosity, TensorboardWriter
from stable_baselines import logger
from stable_baselines import A2C
from stable_baselines.a2c.a2c import A2CRunner
import numpy as np
import gym
import math
from stable_baselines.a2c.utils import discount_with_dones


def custom_discount_with_dones(rewards, dones, gamma, timesteps, future_discounted_reward = 0):
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :param timesteps: ([float]) Time values for each observation, this should be one more than the rewards as it includes both the starting state and the final state.
    :return: ([float]) The discounted rewards
    """
    discounted = []
    ret = future_discounted_reward  # Return: discounted reward
    last_time = timesteps[-1]
    for reward, done, timestep in zip(rewards[::-1], dones[::-1], timesteps[-2::-1]):
        g = math.pow(gamma, last_time - timestep)
        last_time = timestep
        ret = reward + g * ret * (1. - done)  # fixed off by one bug
        discounted.append(ret)
    return discounted[::-1]


class CustomA2C(A2C):
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="A2C"):
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)

            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            runner = CustomA2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)
            self.episode_reward = np.zeros((self.n_envs,))

            t_start = time.time()
            for update in range(1, total_timesteps // self.n_batch + 1):
                # true_reward is the reward without discount
                obs, states, rewards, masks, actions, values, true_reward = runner.run()
                _, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values, update,
                                                                 writer)
                n_seconds = time.time() - t_start
                fps = int((update * self.n_batch) / n_seconds)

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, update * (self.n_batch + 1))

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", update * self.n_batch)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("explained_variance", float(explained_var))
                    logger.dump_tabular()

        return self


class CustomA2CRunner(A2CRunner):
    def run(self):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for _ in range(self.n_steps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            obs, rewards, dones, _ = self.env.step(clipped_actions)
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs2 = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0)
        mb_obs = mb_obs2.reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()

        # Note: Assume the first element in the observation is the timestep
        # This differs from the normal A2CRunner
        timesteps = mb_obs2[:,:,0]
        # Add in the time for the final observation
        timesteps = np.concatenate([timesteps, self.obs[:,0].reshape(timesteps.shape[0], 1)], axis=1)

        # discount/bootstrap off value fn
        for n, (rewards, dones, value, env_timesteps) in enumerate(zip(mb_rewards, mb_dones, last_values, timesteps)):
            env_timesteps = env_timesteps.tolist()
            # timesteps = [i for i in range(len(rewards)+1)]

            rewards = rewards.tolist()
            dones = dones.tolist()

            assert len(dones) == len(rewards)
            assert len(env_timesteps) == len(rewards) + 1

            discounted_rewards = custom_discount_with_dones(rewards, dones, self.gamma, env_timesteps, future_discounted_reward=value)
            mb_rewards[n] = discounted_rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, true_rewards
