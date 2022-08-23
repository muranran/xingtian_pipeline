# (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""Build Atari agent for ppo algorithm."""

import numpy as np
from time import time
from absl import logging
from xt.agent.ppo.ppo import PPO
from xt.agent.ppo.default_config import GAMMA, LAM
from zeus.common.util.register import Registers


@Registers.agent
class AtariPpo3(PPO):
    """Atari Agent with PPO algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.keep_seq_len = True
        self.next_state = None
        self.next_action = None
        self.next_value = None
        self.next_log_p = None
        self.gum=agent_config.get("gum")

    def run_one_episode(self, use_explore, need_collect,*args,**kwargs):
        """
        Do interaction with max steps in each episode.

        :param use_explore:
        :param need_collect: if collect the total transition of each episode.
        :return:
        """
        # clear the old trajectory data
        self.clear_trajectory()
        state = self.env.get_init_state(self.id)

        self._stats.reset()
        cnt=0
        T0=time()
        for _ in range(self.max_step):
            self.clear_transition()
            state = self.do_one_interaction(state, use_explore)
            if need_collect:
                self.add_to_trajectory(self.transition_data)

            if self.transition_data["done"]:
                if not self.keep_seq_len:
                    break
                self.env.reset()
                state = self.env.get_init_state()
        T=time()-T0
        last_pred = self.alg.predict(state)
        return self.get_trajectory(last_pred)
    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        if isinstance(reward,(tuple,list,np.ndarray)):
            pass
        else:
            info.update({'eval_reward': reward})

            self.transition_data.update({"reward": np.sign(reward) if use_explore else reward,
                "done": done,                   
                "info": info
                })
            return self.transition_data

    def handel_predict_value(self, state, predict_val):
        if len(predict_val[0])>1:
            return
            pass
        action = predict_val[0][0]
        logp = predict_val[1][0]
        value = predict_val[2][0]

        # update transition data
        self.transition_data.update({
            'cur_state': state,
            'action': action,
            'logp': logp,
            'value': value,
        })

        return action
    def get_random_state(self,gum):
        return [self.env.step(self.env.env.action_space.sample())[0] for i in range(gum)]
    def do_one_interaction(self, raw_state, use_explore=True):
        """
        Use the Agent do one interaction.

        User could re-write the infer_action and handle_env_feedback functions.
        :param raw_state:
        :param use_explore:
        :return:
        """
        _start0 = time()
        action = self.infer_action(raw_state, use_explore)
        self._stats.inference_time += time() - _start0


        #raw_state_list = [raw_state for i in range(self.gum)]
        raw_state_list=self.get_random_state(self.gum)
        _start1 = time()
        action_list = self.infer_action(raw_state_list, use_explore)
        self._stats.env_step_time += time() - _start1

        #_start1 = time()
        next_raw_state, reward, done, info = self.env.step(action, self.id)
        #self._stats.env_step_time += time() - _start1
        self._stats.iters += 1

        self.handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        return next_raw_state
