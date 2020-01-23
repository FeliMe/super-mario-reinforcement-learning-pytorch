"""
Some parts are from:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQNAgent:
    """ DQN agent """

    def __init__(self, actions, max_memory, device, save_dir='models/',
                 continue_training=False, model_path=None, double_q=True):
        self.actions = actions
        self.save_dir = save_dir
        self.device = device
        self.double_q = double_q
        self.batch_size = 32
        self.gamma = 0.9
        self.eps = 1
        self.eps_decay = 0.99999975
        self.eps_min = 0.1
        self.target_update = 10
        self.step = 0
        self.n_update_target = 10000
        self.save_each = 100000
        self.learn_each = 3
        self.learn_step = 0
        self.burnin = 100000

        self.policy_net = dqn(actions).to(device)
        self.target_net = dqn(actions).to(device)

        if continue_training:
            self.step = int(model_path.split('/')[1].split('.')[0].split('model')[1])
            print(f"Continuing from step {self.step}")
            self.policy_net.load_state_dict(torch.load(model_path))

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(max_memory)

        self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.criterion = nn.SmoothL1Loss()

        self.state_dict = copy.deepcopy(self.policy_net.state_dict())

    def update_target_network(self):
        """ Copy weights to target network """
        print('Syncing policy net and target net')
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """ Saves current model to disk """
        self.state_dict = copy.deepcopy(self.policy_net.state_dict())
        os.makedirs(self.save_dir, exist_ok=True)
        print('Saving model to {}'.format(
            os.path.join(self.save_dir, f'model{self.step}.pt')))
        torch.save(self.state_dict,
                   os.path.join(self.save_dir, f'model{self.step}.pt'))

    def select_action(self, state):
        """ Select action """
        sample = random.random()
        self.eps *= self.eps_decay
        self.eps = max(self.eps_min, self.eps)
        self.step += 1
        if sample > self.eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.actions)]],
                                device=self.device, dtype=torch.long)

    def optimize_model(self):
        """ Gradient descent """
        # Sync target network
        if self.step % self.n_update_target == 0:
            self.update_target_network()
        # Checkpoint model
        if self.step % self.save_each == 0:
            self.save_model()
        # Break if burn-in
        if self.step < self.burnin:
            return
        if self.step == self.burnin:
            print("Start Learning")
        # Break if no training
        if self.learn_step < self.learn_each:
            self.learn_step += 1
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043
        # for detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                          if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been
        # taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed
        # based on the "older" target_net; selecting their best reward with
        # max(1)[0]. This is merged based on the mask, such that we'll have
        # either the expected state value or 0 in case the state was final.
        next_q = self.target_net(non_final_next_states).detach()
        if self.double_q:
            a = self.policy_net(
                non_final_next_states).max(1)[1].detach()
            next_q_unmasked = next_q[torch.arange(0, self.batch_size), a]
        else:
            next_q_unmasked = next_q.max(1)[0]

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = next_q_unmasked

        # next_state_values = torch.zeros(self.batch_size, device=self.device)
        # next_state_values[non_final_mask] = self.target_net(
        #     non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.learn_step = 0


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def dqn(actions):
    return nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, actions)
    )


def convert_state_to_tensor(state):
    state = np.array(state).transpose((2, 0, 1))
    state = np.ascontiguousarray(state, dtype=np.float32) / 255.
    state = torch.from_numpy(state)
    return state.unsqueeze(0)
