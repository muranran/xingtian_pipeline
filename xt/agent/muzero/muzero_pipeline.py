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
"""Build MuZero agent."""
from collections import defaultdict
from time import time, sleep

import numpy as np

from xt.agent.agent import Agent
from xt.agent.muzero.default_config import NUM_SIMULATIONS, GAMMA, TD_STEP
from zeus.common.ipc.message import message, set_msg_info
from zeus.common.util.register import Registers
from zeus.common.util.common import import_config
import math
from xt.agent.muzero.default_config import PB_C_BASE, PB_C_INIT
from xt.agent.muzero.default_config import ROOT_DIRICHLET_ALPHA
from xt.agent.muzero.default_config import ROOT_EXPLORATION_FRACTION
from xt.agent.muzero.default_config import GAMMA
from xt.agent.muzero.util import MinMaxStats, Node, soft_max_sample
from collections import OrderedDict


class Mcts(object):
    """MCTS operation."""

    def __init__(self, agent, root_state, nums=10):
        self.network = agent.alg.actor
        self.action_dim = agent.alg.action_dim
        self.num_simulations = agent.num_simulations
        self.min_max_stats = [MinMaxStats(None) for i in range(nums)]
        self.discount = GAMMA
        self.actions = [range(self.action_dim) for i in range(nums)]
        self.pb_c_base = PB_C_BASE
        self.pb_c_init = PB_C_INIT
        self.root_dirichlet_alpha = ROOT_DIRICHLET_ALPHA
        self.root_exploration_fraction = ROOT_EXPLORATION_FRACTION
        self.nums = nums
        self.root = [Node(0) for i in range(nums)]
        # root_state = root_state[0].reshape((1,) + root_state[0].shape)
        root_state = [root_state[i] for i in range(nums)]  # [i].reshape((1,) + root_state[i].shape)
        network_output = self.network.initial_inference(root_state)

        for i in range(nums):
            self.init_node(self.root[i], network_output[i], i)

    def init_node(self, node, network_output, index):
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward

        policy = [p for p in network_output.policy]
        # print(self.actions[index])
        for action in self.actions[index]:
            # assert action not in node.children
            node.children[action] = Node(policy[action])

    def backpropagate(self, search_path, value, index):
        """Propagate the evaluation all the way up the tree to the root at the end of a simulation."""

        # print("back://", search_path.__len__(), value)

        for node in search_path[::-1]:
            node.value_sum += value
            node.visit_count += 1
            self.min_max_stats[index].update(node.value())

            value = node.reward + self.discount * value

    def backpropagate_batch(self, search_path, value, index):
        for i in range(self.nums):
            self.backpropagate(search_path[i], value[i], i)

    def run_mcts(self):
        """
        Run Core Monte Carlo Tree Search algorithm.

        To decide on an action, we run N simulations, always starting at the root of
        the search tree and traversing the tree according to the UCB formula until we
        reach a leaf node.
        """

        nums = self.nums
        for _ in range(self.num_simulations):
            node = self.root
            interval_node = OrderedDict()

            search_path = [[node[i]] for i in range(nums)]
            history = [[] for i in range(nums)]

            for i in range(nums):
                tmp_node = node[i]
                interval_node[i] = tmp_node
                cnt = 0
                while tmp_node.expanded():
                    action, tmp_node = self.select_child(tmp_node, i)
                    interval_node[i] = tmp_node

                    search_path[i].append(tmp_node)
                    history[i].append(action)
                    cnt += 1

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = []
            for i in range(nums):
                parent.append(search_path[i][-2])

            network_output = self.network.recurrent_inference(np.array([p.hidden_state for p in parent]),
                                                              np.array([h[-1] for h in history]))

            for i in range(nums):
                self.init_node(interval_node[i], network_output[i], i)

                self.backpropagate(search_path[i], network_output[i].value, i)

    def select_action(self, mode='softmax'):
        """
        Select action.

        After running simulations inside in MCTS, we select an action based on the root's children visit counts.
        During training we use a softmax sample for exploration.
        During evaluation we select the most visited child.
        """
        node = self.root
        nums = self.nums
        batch_action = []
        for i in range(nums):
            visit_counts = [child.visit_count for child in node[i].children.values()]
            # print("=================={}====================".format(visit_counts))
            actions = self.actions[i]
            action = None
            if mode == 'softmax':
                action = soft_max_sample(visit_counts, actions, 1)
            elif mode == 'max':
                action = np.argmax(visit_counts)
            batch_action.append(action)
        return batch_action

    def ucb_score(self, parent, child, index):
        """
        Calculate UCB score.

        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = self.min_max_stats[index].normalize(child.value())
        else:
            value_score = 0
        return prior_score + value_score

    def add_exploration_noise(self, node, index):
        actions = self.actions[index]
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * self.action_dim)
        frac = self.root_exploration_fraction
        for i, _noise in zip(actions, noise):
            node.children[i].prior = node.children[i].prior * (1 - frac) + _noise * frac

    def get_info(self, index):
        """Get train info from mcts tree."""
        child_visits = [self.root[index].children[a].visit_count for a in self.actions[index]]
        sum_visits = sum(child_visits)
        child_visits = [visits / sum_visits for visits in child_visits]
        return {"child_visits": child_visits, "root_value": self.root[index].value()}

    def select_child(self, node, index):
        """Select the child with the highest UCB score."""
        _, action, child = max(
            (self.ucb_score(node, child, index), action, child) for action, child in node.children.items())
        return action, child


@Registers.agent
class Muzero(Agent):
    """Build Agent with Muzero algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        import_config(globals(), agent_config)
        super().__init__(env, alg, agent_config, **kwargs)
        self.num_simulations = NUM_SIMULATIONS
        self.nums = kwargs.get("env_num")
        self.trajectory = [defaultdict(list) for i in range(self.nums)]
        self.env_num = kwargs.get("env_num")
        self.env_ids = np.array(list(range(self.env_num)))
        self.lock = kwargs.get("lock")
        self.group_num = kwargs.get("group_num")
        self.group_id = kwargs.get("group_id")
        self.first = True
        self.next_init_trajectory = []
        self.using_lock = False

    def agent_network_initial_inference(self, state):
        infer_output = self.alg.actor.initial_inference(state)
        return infer_output[0]

    def infer_action(self, state, use_explore):
        """
        Infer action.

        We then run a Monte Carlo Tree Search using only action sequences and the
        model learned by the networks.
        """
        nums = self.nums
        mcts = Mcts(self, state, nums=len(self.env_ids))
        if use_explore:

            for i, env_id in enumerate(self.env_ids):
                mcts.add_exploration_noise(mcts.root[i], i)

        mcts.run_mcts()
        action = mcts.select_action()
        for i, env_id in enumerate(self.env_ids):
            self.transition_data[env_id].update({"cur_state": state[i], "action": action[i]})
            self.transition_data[env_id].update(mcts.get_info(i))
        # for i in range(nums):
        #     self.transition_data[i].update({"cur_state": state[i], "action": action[i]})
        #     self.transition_data[i].update(mcts.get_info(i))

        return action

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

    # def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
    #     nums = self.nums
    #     # fixme
    #     for i in range(nums):
    #         # info.update({'eval_reward': reward})
    #         _info = {'eval_reward': reward[i]}
    #         self.transition_data[i].update({
    #             "reward": reward[i],
    #             "done": done[i],
    #             "info": _info
    #         })
    #
    #     return self.transition_data

    def get_trajectory(self):
        nums = self.nums
        trajectory = []
        for env_id in set(np.arange(self.env_num)) - set(self.env_ids):
            self.next_init_trajectory.append(
                {
                    "env_id": env_id,
                    "data": {
                        "child_visits": self.trajectory[env_id]["child_visits"][-1],
                        "root_value": self.trajectory[env_id]["root_value"][-1],
                        "cur_state": self.trajectory[env_id]["cur_state"][-1],
                        "action": self.trajectory[env_id]["action"][-1]
                    }
                }
            )
        for i in range(nums):
            self.data_proc(i)
            traj = message(self.trajectory[i].copy())
            set_msg_info(traj, agent_id=self.id)
            trajectory.append(traj)

        return trajectory

    def data_proc(self, index):
        traj = self.trajectory[index]
        value = traj["root_value"]
        reward = traj["reward"]
        dones = np.asarray(traj["done"])
        discounts = ~dones * GAMMA

        target_value = [reward[-1]] * len(reward)
        for i in range(len(reward) - 1):
            end_index = min(i + TD_STEP, len(reward) - 1)
            sum_value = value[end_index]

            for j in range(i, end_index)[::-1]:
                sum_value = reward[j] + discounts[j] * sum_value

            target_value[i] = sum_value

        self.trajectory[index]["target_value"] = target_value

    def sync_model(self):
        ret_model_name = None
        while True:
            model_name = self.recv_explorer.recv(name=None, block=False)
            if model_name:
                ret_model_name = model_name
            else:
                break

        return ret_model_name

    # def clear_trajectory(self):
    #     for trj in self.trajectory:
    #         trj.clear()
    def clear_trajectory(self):
        for trj in self.trajectory:
            trj.clear()
        for trans_data in self.next_init_trajectory:
            for k, v in trans_data["data"].items():
                self.trajectory[trans_data["env_id"]][k].append(v)
        self.next_init_trajectory.clear()

    def add_to_trajectory2(self, transition_data, tjid=None):
        # print("type {}/n{}".format(tjid,transition_data))
        if tjid != None:
            for k, val in transition_data.items():
                self.trajectory[tjid][k].append(val)
        else:
            for k, val in transition_data.items():
                self.trajectory[k].append(val)

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

        if self.using_lock:
            self.lock[self.group_id].acquire(True)

        _start1 = time()
        next_raw_state, reward, done, info = self.env.step(action, self.id)
        self._stats.env_step_time += time() - _start1
        self._stats.iters += 1

        if self.using_lock:
            next_gid = self.group_id + 1 if self.group_id < self.group_num - 1 else 0
            self.lock[next_gid].release()

        self.handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        return next_raw_state

    def run_one_episode(self, use_explore, need_collect):
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
        self.transition_data = [defaultdict() for i in range(self.env_num)]
        for _ in range(self.max_step):
            self.clear_transition()
            state = self.do_one_interaction(state, use_explore)

            if need_collect:
                # self.add_to_trajectory2(self.transition_data)

                for i, transition in enumerate(self.transition_data):
                    if transition:
                        self.add_to_trajectory2(transition, i)

            # if self.transition_data["done"]:
            #     if not self.keep_seq_len:
            #         break
            #     self.env.reset()
            #     state = self.env.get_init_state()
        # for env_id in set(list(range(self.nums))) - set(self.env_ids):
        #     for k in ["child_visits", "root_value", "cur_state", "action"]:
        #         self.trajectory[env_id][k].pop()

        return self.get_trajectory()


@Registers.agent
class MuzeroAtari2(Muzero):
    """ Agent with Muzero algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.num_simulations = NUM_SIMULATIONS
        self.keep_seq_len = True
        self.env_num = kwargs.get("env_num")

    def infer_action(self, state, use_explore):
        """
        We then run a Monte Carlo Tree Search using only action sequences and the
        model learned by the networks.
        """
        state = [s.astype('uint8') for s in state]
        action = super().infer_action(state, use_explore)

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        next_raw_state = np.array(next_raw_state).astype('uint8')
        super().handle_env_feedback(next_raw_state, reward, done, info, use_explore)

        return self.transition_data


if __name__ == '__main__':
    agent = MuzeroAtari2(None, None, None)
