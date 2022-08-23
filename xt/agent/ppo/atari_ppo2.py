# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""Build Atari agent for ppo algorithm."""
from copy import deepcopy

import numpy as np
from time import time, sleep
from zeus.common.ipc.message import message, set_msg_info
from absl import logging
from xt.agent.ppo.ppo import PPO
from xt.agent.ppo.default_config import GAMMA, LAM
from zeus.common.util.register import Registers
from collections import defaultdict
import logging


@Registers.agent
class AtariPpo2(PPO):
    """Atari Agent with PPO algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.keep_seq_len = True
        self.next_state = None
        self.next_action = None
        self.next_value = None
        self.next_log_p = None
        self.env_num = kwargs.get("env_num")
        self.trajectory = [defaultdict(list) for i in range(self.env_num)]  # env_num
        self.env_ids = np.array(list(range(self.env_num)))
        self.lock = kwargs.get("lock")
        self.group_num = kwargs.get("group_num")
        self.group_id = kwargs.get("group_id")
        self.first = True
        # print("============================", self.env_num, self.group_num, self.group_id)
        self.next_init_trajectory = []
        self.using_lock = True

    def do_one_interaction(self, raw_state, use_explore=True):
        _start0 = time()
        action_list = self.infer_action(raw_state, use_explore)
        self._stats.inference_time += time() - _start0

        if self.using_lock:
            self.lock[self.group_id].acquire(True)

        _start1 = time()
        next_raw_state, reward, done, info = self.env.step(action_list)
        self._stats.env_step_time += time() - _start1
        self._stats.iters += 1

        if self.using_lock:
            next_gid = self.group_id + 1 if self.group_id < self.group_num - 1 else 0
            self.lock[next_gid].release()

        self.handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        return next_raw_state

    def clear_trajectory(self):
        for trj in self.trajectory:
            trj.clear()
        for trans_data in self.next_init_trajectory:
            for k, v in trans_data["data"].items():
                self.trajectory[trans_data["env_id"]][k].append(v)
        self.next_init_trajectory.clear()

    def clear_transition(self):
        for transition in self.transition_data:
            transition.clear()

    def run_one_episode(self, use_explore, need_collect):
        # clear the old trajectory data
        self.clear_trajectory()
        state = self.env.get_init_state(self.id)
        self._stats.reset()
        cnt = 0
        T0 = time()
        self.transition_data = [defaultdict() for i in range(self.env_num)]
        for _ in range(int(self.max_step)):
            self.clear_transition()
            state = self.do_one_interaction(state, use_explore)
            cnt += 1
            if need_collect:
                for i, transition in enumerate(self.transition_data):
                    if transition:
                        self.add_to_trajectory2(transition, i)

        T = time() - T0
        last_pred = self.alg.predict(state)
        return self.get_trajectory(last_pred)

    def handel_predict_value(self, state, predict_val):
        action = predict_val[0]
        logp = predict_val[1]
        value = predict_val[2]

        # update transition data
        for i, env_id in enumerate(self.env_ids):
            self.transition_data[env_id].update({
                'cur_state': state[i],
                'action': action[i],
                'logp': logp[i],
                'value': value[i],
            })

        return action

    def add_to_trajectory2(self, transition_data, tjid=None):
        if tjid != None:
            for k, val in transition_data.items():
                self.trajectory[tjid][k].append(val)
        else:
            for k, val in transition_data.items():
                self.trajectory[k].append(val)

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        if isinstance(reward, (tuple, list, np.ndarray)):
            self.env_ids = self.env.finished_env
            for i, _info in enumerate(info):
                info[i].update({'eval_reward': reward[i]})
                self.transition_data[_info["env_id"]].update({
                    "reward": np.sign(reward[i]) if use_explore else reward[i],
                    "done": done[i],
                    "info": info[i]
                })
        else:
            info.update({'eval_reward': reward})
            self.transition_data[0].update({
                "reward": np.sign(reward) if use_explore else reward,
                "done": done,
                "info": info
            })

        return self.transition_data

    def data_proc2(self, index):
        """Process data."""
        traj = self.trajectory[index]

        # print(len(traj["reward"]),len(traj["value"])-1,sll)
        state = np.asarray(traj['cur_state'])
        action = np.asarray(traj['action'])
        logp = np.asarray(traj['logp'])
        value = np.asarray(traj['value'])
        reward = np.asarray(traj['reward'])
        done = np.asarray(traj['done'])

        # print(
        #     "env_{}_{} : state {} | action {} | logp {} | value {} | reward {} | done {}\nfinished env : {}\ninfo : {}". \
        #         format(self.group_id, index, state.shape[0], action.shape[0], logp.shape[0], value.shape[0],
        #                reward.shape[0], done.shape[0], sorted(self.env_ids), traj['info'][0]))

        tr = np.sum(reward)

        next_value = value[1:]
        value = value[:-1]
        done = np.expand_dims(done, axis=1)
        reward = np.expand_dims(reward, axis=1)
        discount = ~done * GAMMA
        delta_t = reward + discount * next_value - value
        adv = delta_t

        for j in range(len(adv) - 2, -1, -1):
            adv[j] += adv[j + 1] * discount[j] * LAM

        self.trajectory[index]['cur_state'] = state
        self.trajectory[index]['action'] = action
        self.trajectory[index]['logp'] = logp
        self.trajectory[index]['adv'] = adv
        self.trajectory[index]['old_value'] = value
        self.trajectory[index]['target_value'] = adv + value

        del self.trajectory[index]['value']

    def get_trajectory(self, last_pred=None):
        """Get trajectory"""
        # Need copy, when run with explore time > 1,
        # if not, will clear trajectory before sent.
        last_val = last_pred[2]

        trajectory = []
        for i, env_id in enumerate(self.env_ids):
            self.trajectory[env_id]['value'].append(last_val[i])
        for env_id in set(np.arange(self.env_num)) - set(self.env_ids):
            self.next_init_trajectory.append({"env_id": env_id,
                                              "data": {
                                                  "cur_state": self.trajectory[env_id]['cur_state'].pop(),
                                                  "action": self.trajectory[env_id]['action'].pop(),
                                                  "logp": self.trajectory[env_id]['logp'].pop(),
                                                  "value": self.trajectory[env_id]['value'][-1]
                                              }})
        for i, trj in enumerate(self.trajectory):
            self.data_proc2(i)
            tmp = message(self.trajectory[i])
            set_msg_info(tmp, agent_id=i)
            trajectory.append(tmp)
        # trajectory = message(deepcopy(self.trajectory))

        return deepcopy(trajectory)
