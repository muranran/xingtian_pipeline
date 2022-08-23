# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Make atari env for simulation."""
import random
import time

import cv2
import numpy as np
from collections import deque

from xt.environment.environment import Environment
from xt.environment.gym import infer_action_type
from xt.environment.gym.atari_wrappers import make_atari
from zeus.common.util.register import Registers
import envpool

cv2.ocl.setUseOpenCL(False)


@Registers.env
class AtariEnv(Environment):
    """Encapsulate an openai gym environment."""

    def init_env(self, env_info):
        """
        Create a atari environment instance.

        :param: the config information of environment.
        :return: the instance of environment
        """
        env = make_atari(env_info)

        self.dim = env_info.get('dim', 84)
        self.action_type = infer_action_type(env.action_space)

        self.init_obs = np.zeros((self.dim, self.dim, 1), np.uint8)
        self.stack_size = 4
        self.stack_obs = deque(maxlen=4)

        self.init_stack_obs(self.stack_size)
        self.init_state = None
        self.done = True

        return env

    def init_stack_obs(self, num):
        for _ in range(num):
            self.stack_obs.append(self.init_obs)

    def reset(self):
        # print('[AtariEnv] reset()')
        """
        Reset the environment, if done is true, must clear obs array.

        :return: the observation of gym environment
        """
        if self.done:
            obs = self.env.reset()
            self.init_stack_obs(self.stack_size - 1)
            self.stack_obs.append(self.obs_proc(obs))

        state = np.concatenate(self.stack_obs, -1)
        self.init_state = state
        return state

    def step(self, action, agent_index=0):
        # print('[AtariEnv] step()')
        """
        Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (state, reward, done, info).

        :param action: action
        :param agent_index: the index of agent
        :return: state, reward, done, info
        """
        obs, reward, done, info = self.env.step(action)
        # print('debug-1', obs.shape)
        if done:
            self.init_stack_obs(self.stack_size - 1)

        self.stack_obs.append(self.obs_proc(obs))
        # self.stack_obs.append(obs)
        state = np.concatenate(self.stack_obs, -1)
        self.done = done
        # print('debug-2', state.shape)
        return state, reward, done, info

    def obs_proc(self, obs):
        # convert the RGB img to Gray img
        # shape: (width, height, 3)  -->  (dim, dim, 1)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.dim, self.dim),
                         interpolation=cv2.INTER_AREA)
        obs = np.expand_dims(obs, -1)
        return obs


@Registers.env
class SinglePool(Environment):
    def init_env(self, env_info):
        print('[GGLC] SinglePool created')
        self._env = envpool.make(
            env_info['name'].replace('NoFrameskip-v4', '-v5'),
            env_type='gym',
            num_envs=1,  # d1
            frame_skip=4,  # d4
            episodic_life=True,  # df
            stack_num=4,  # d
            noop_max=30,  # d, noop_action->0
            seed=random.randint(0, 1000),  # d42
        )
        self.last_state = np.zeros((84, 84, 4))

        # 不写会导致 infer(agent.py: do_one_interaction) 的 state 为 None。报错
        self.init_state = None

        # 不写会报 NotImplementError (tf_dist.py#139)
        gym_env = make_atari(env_info)
        self.dim = env_info.get('dim', 84)
        self.action_type = infer_action_type(gym_env.action_space)
        return gym_env

    def init_stack_obs(self, num):
        pass

    def reset(self):
        self.init_state = self.last_state
        return self.last_state

    def step(self, action: int, agent_index=0):
        obs, reward, done, info = self._env.step(np.array([action]))
        obs = obs.transpose(0, 2, 3, 1)
        _info = {'real_done': info['lives'][0] == 0, 'eval_reward': info['reward'][0], 'ale.lives': info['lives'][0]}
        self.last_state = obs[0]
        return obs[0], reward[0], done[0], _info

    def obs_proc(self, obs):
        pass


@Registers.env
class VectorAtariEnv(Environment):
    """Vectorize atari environment to speedup."""

    def init_env(self, env_info):
        """Create multi-env as a vector."""
        self.vector_env_size = env_info.get("vector_env_size")
        assert self.vector_env_size is not None, "vector env must assign 'env_num'."

        self.env_vector = list()
        for _ in range(self.vector_env_size):
            self.env_vector.append(AtariEnv(env_info))

        self.step_count = 0
        self.epi_count = 0
        self.len_count = 0

    def reset(self):
        """Reset each env within vector."""
        state = [env.reset() for env in self.env_vector]
        self.init_state = state
        # return: list()
        #   len: vector_size
        #   state(obs): np.ndarray(dim, dim, 4)
        return state

    def step(self, action, agent_index=0):
        """
        Step in order.

        :param action:
        :param agent_index:
        :return:
        """
        batch_obs, batch_reward, batch_done, batch_info = list(), list(), list(), list()
        # print('##############', len(self.env_vector), self.vector_env_size, action)
        # self.step_count += 1
        # self.len_count += 1
        for env_id in range(self.vector_env_size):
            obs, reward, done, info = self.env_vector[env_id].step(action[env_id])
            if done:
                # self.epi_count += 1
                # print('[episode] step={} episode={} len={}'.format(self.step_count, self.epi_count, self.len_count))
                # self.len_count = 0
                obs = self.env_vector[env_id].reset()

            batch_obs.append(obs)
            batch_reward.append(reward)
            batch_done.append(done)
            batch_info.append(info)

        # print('[Vector] step return batch(obs, rew, done, info): len-{} obs-{} rew-{} done-{} info-{}'.format(
        #     len(batch_obs), batch_obs[0].shape, batch_reward, batch_done, batch_info[0]
        # ))

        # return:
        #   batch_size = vector_size
        #   obs: (84, 84, 4)
        #   rew: float
        #   done: bool
        #   info: dict**  {'ale.lives': int, 'real_done': bool}
        return batch_obs, batch_reward, batch_done, batch_info

    def get_env_info(self):
        """
        Return environment's basic information.

        vector environment only support single agent now.
        """
        self.reset()
        env_info = {
            "n_agents": self.n_agents,
            "api_type": self.api_type,
        }
        agent_ids = [0]
        env_info.update({"agent_ids": agent_ids})

        # print('============================\n[Vector] get_env_info return {}'.format(env_info))
        # return:
        #   {'n_agents': 1, 'api_type': 'standalone', 'agent_ids': [0]}
        return env_info

    def close(self):
        [env.close() for env in self.env_vector]


@Registers.env
class EnvPool(Environment):
    """use envpool to speedup."""

    def init_env(self, env_info):
        print('[GGLC] EnvPool created')
        self.size = env_info.get("size")
        self.name = env_info.get("name")
        self.name = self.name.replace('NoFrameskip-v4', '-v5')
        assert self.size is not None and self.name is not None, "envpool must assign 'name' and 'size'."
        # self.batch_size = env_info.get("batch_size", self.size)

        self.pool = envpool.make(
            task_id=self.name,
            env_type='gym',
            num_envs=self.size,  # d1
            # batch_size  =   self.batch_size,  # den
            # num_threads=self.size,  # dbs
            frame_skip=4,  # d4
            episodic_life=True,  # df
            stack_num=4,  # d
            noop_max=30,  # d, noop_action->0
            seed=random.randint(0, 10000),  # d42
            # thread_affinity_offset=-1,  # d-1
            # repeat_action_probability=.0,  # d
            # max_episode_steps=108000,  # d
        )

        # fake reset, just return last state
        self.dim = env_info.get('dim', 84)
        self.last_state = np.zeros((self.size, self.dim, self.dim, 4))

        # for ppo, see SinglePool
        self.init_state = None
        gym_env = make_atari(env_info)
        self.action_type = infer_action_type(gym_env.action_space)

        # for log (episode length)
        self.len_count = np.zeros(self.size)  # count the length of episode
        self.step_count = 0  # count step
        self.epi_count = 0

        return gym_env

    def reset(self):
        self.init_state = self.last_state
        # return: list()
        #   len: vector_size
        #   state(obs): np.ndarray(dim, dim, 4)
        return self.last_state

    def step(self, action, agent_index=0):
        """
        :param action: list() or np.array()
        :param agent_index:
        :return: (list(), list(), list(), list())
            obs: list(np.ndarray(dim, dim, 4))
            rew: list(float)
            done: list(bool)
            info: list(dict{'ale.lives':int, 'real_done':bool})
        """
        _start = time.time()
        obs, rew, done, info = self.pool.step(np.array(action))
        obs = obs.transpose(0, 2, 3, 1)
        self.last_state = obs  # fixme: useless

        _info = []
        for env_id in range(self.size):
            _info.append({'real_done': info['lives'][env_id] == 0 and done[env_id], 'ale.lives': info['lives'][env_id]})

        return list(obs), list(rew), list(done), _info

        # return:
        #   batch_size = vector_size
        #   obs: (84, 84, 4)
        #   rew: float
        #   done: bool
        #   info: dict**  {'ale.lives': int, 'real_done': bool}

        # print('[INFO] EnvPool action: {}'.format(action))
        # print('[INFO] EnvPool state: {}'.format(done))
        # self.step_count += 1
        # for i in range(self.size):
        #     self.len_count[i] += 1
        #     if done[i]:
        #         self.epi_count += 1
        #         print('[episode] step={} episode={} len={}'.format(self.step_count, self.epi_count, self.len_count[i]))
        #         self.len_count[i] = 0

        # print('debug', action, done)
        # print('envpool.step(): {:.2f} ms'.format(1000*(time.time()-_start)))

    def get_env_info(self):
        """
        Return environment's basic information.
        vector environment only support single agent now.
        """
        self.reset()
        env_info = {
            "n_agents": 1,
            "api_type": 'standalone',
            "agent_ids": [0]
        }
        return env_info

    def close(self):
        self.pool.close()
