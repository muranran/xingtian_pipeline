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

import numpy as np
from time import time,sleep
from zeus.common.ipc.message import message, set_msg_info
from absl import logging
from xt.agent.ppo.ppo import PPO
from xt.agent.ppo.default_config import GAMMA, LAM
from zeus.common.util.register import Registers
from collections import defaultdict
import logging
@Registers.agent
class AtariPpo20(PPO):
    """Atari Agent with PPO algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.keep_seq_len = True
        self.next_state = None
        self.next_action = None
        self.next_value = None
        self.next_log_p = None
        self.env_num=agent_config.get("gum")
        #self.env2=env[1]
        self.trajectory = [defaultdict(list) for i in range(self.env_num)] # env_num
    
    
    def do_one_interaction(self, raw_state, use_explore=True,lock=None,gid=0,gn=0):
        
        raw_state_list = raw_state
        _start0=time()

        action_list = self.infer_action(raw_state_list, use_explore)
        if raw_state is None or self.first:
            #action_list=self.get_random_act(self.env_num+4)
            self.first=False
        self._stats.inference_time += time() - _start0
        #print("*"*10,"action batch test.",action,action_list,"infer time is {}".format(t1-_start0),"batch infer time is {}".format(time.time()-t1))
        #print("=============waiting gid {}===================".format(gid))
        L0=time()
        lock[gid].acquire(True)
        L=time()-L0
        next_gid=gid+1 if gid< gn-1 else 0
        #lock[next_gid].release()
        #sleep(1)
        if not hasattr(self,"env_id"):
            #action_list=np.append(action_list,[0,0])
            #self.env_id=list(range(0,self.env_num+4))
            pass
        _start1=time()
       # print("===========================",len(self.env_id),len(action_list))
        #self.env_id=list(range(0,26))
        #next_raw_state, reward, done, info = self.env2.step(action_list,np.array(self.env_id))
        next_raw_state, reward, done, info = self.env2.step(action_list)
        self.env_id=list(set(info["env_id"]))
        #print(info["env_id"])
        self._stats.env_step_time += time() - _start1
        lock[next_gid].release()
        next_raw_state=list(map(lambda x:x.reshape(84,84,4),next_raw_state))
        self._stats.iters += 1

        self.handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        return next_raw_state
        
    def get_env_init_state(self):
        self.env2.reset()
        obs,rew,done,info=self.env2.step(np.array([self.env.env.action_space.sample() for i in range(self.env_num)]),np.array(list(range(0,self.env_num))))
        return list(map(lambda x:x.reshape((84,84,4)),obs))


    def get_random_act(self,num):
        return np.array([self.env.env.action_space.sample() for i in range(num)])
    
    
    def clear_trajectory(self):
        for trj in self.trajectory:
            trj.clear()
    
    
    def run_one_episode(self, use_explore, need_collect,env2,lock,gid,gn):
        # clear the old trajectory data
        if not hasattr(self,"env2"):
            self.env2=env2
        self.clear_trajectory()
        state=self.get_env_init_state()
        #state=None
        self.first=False
        self._stats.reset()
        cnt=0
        T0=time()
        self.transition_data=[defaultdict() for i in range(self.env_num)]
        for _ in range(int(self.max_step)):
            state = self.do_one_interaction(state, use_explore,lock,gid,gn)
            cnt+=1
            if need_collect:
                for i,transition_data in enumerate(self.transition_data):
                    self.add_to_trajectory(transition_data,i)
        T=time()-T0
        last_pred = self.alg.predict(state)
        return self.get_trajectory(last_pred)
    
    
    def handel_predict_value(self, state, predict_val):    
        action = predict_val[0]
        logp = predict_val[1]
        value = predict_val[2]

        # update transition data
        for i in range(len(self.transition_data)):
            self.transition_data[i].update({
                'cur_state': state[i],
                'action': action[i],
                'logp': logp[i],
                'value': value[i],
            })

        return action
           

    def add_to_trajectory(self, transition_data,tjid=None):
        if tjid!=None:
            for k, val in transition_data.items():
                self.trajectory[tjid][k].append(val)
        else:
            for k, val in transition_data.items():
                self.trajectory[k].append(val)

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        if isinstance(reward,(tuple,list,np.ndarray)):
            if not hasattr(self,"lives"):
                self.lives=[5 for i in range(self.env_num)]
            for i in range(len(self.transition_data)):
                dddd=done[i]
                lives=info["lives"][i]
                if lives<self.lives[i] and lives>0:
                    dddd=True
                    self.lives[i]=lives
                self.transition_data[i].update({
                    "reward": np.sign(reward[i]) if use_explore else reward[i],
                    "done":dddd,
                    "info": {"real_done":done[i],"eval_reward":reward[i]}
                })
            #self.lives=info["lives"].copy()
        else:
            info.update({'eval_reward': reward})

            self.transition_data.update({
                "reward": np.sign(reward) if use_explore else reward,
                "done": done,
                "info": info
            })

        return self.transition_data

    def data_proc2(self,index):
        """Process data."""
        traj = self.trajectory[index]
        state = np.asarray(traj['cur_state'])
        action = np.asarray(traj['action'])
        logp = np.asarray(traj['logp'])
        value = np.asarray(traj['value'])
        reward = np.asarray(traj['reward'])
        done = np.asarray(traj['done'])
        
        tr=np.sum(reward)
        if tr>0:
            pass
            #print("=========================REWARD \n{}\n=====================\n=================================DONE \n{}\n====================================".format(reward,[int(d) for d in done]))
        
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


        trajectory=[]
        for i,tj in enumerate(self.trajectory):
            self.trajectory[i]['value'].append(last_val[i])
            self.data_proc2(i)
            tmp=message(self.trajectory[i].copy())
            set_msg_info(tmp, agent_id=i)
            trajectory.append(tmp)
        # trajectory = message(deepcopy(self.trajectory))


        return trajectory
