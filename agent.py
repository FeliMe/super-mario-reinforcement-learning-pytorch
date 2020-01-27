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
                        ('state', 'action', 'next_state', 'reward', 'done'))


class DQNAgent:
    """ DQN agent """

    def __init__(self, actions, max_memory, device, save_dir='models/',
                 continue_training=False, model_path=None, double_q=True):
        self.actions = actions
        self.save_dir = save_dir
        self.device = device
        self.double_q = double_q
        self.batch_size = 32
        self.burnin = 10000
        self.gamma = 0.99
        self.exploration = LinearSchedule(1000000, 0.1)
        self.eps = self.exploration.value(0.0)
        self.step = 0
        self.n_update_target = 1000
        self.save_each = 100000
        self.learn_each = 3
        self.learn_step = 0

        self.policy_net = dqn(actions).to(device)
        self.target_net = dqn(actions).to(device)

        if continue_training:
            self.step = int(model_path.split('/')[1].split('.')[0].split('model')[1])
            print(f"Continuing from step {self.step}")
            self.policy_net.load_state_dict(torch.load(model_path))

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(max_memory)

        self.opt = torch.optim.RMSprop(
            self.policy_net.parameters(), lr=0.0001, alpha=0.95, eps=0.01)

        self.state_dict = copy.deepcopy(self.policy_net.state_dict())

    def update_target_network(self):
        """ Copy weights to target network """
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
        self.eps = self.exploration.value(self.step)
        self.step += 1
        if sample > self.eps:
            with torch.no_grad():
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
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        next_state_batch = torch.cat([s for s in batch.next_state
                                      if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        not_done_mask = (1 - done_batch.int())

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been
        # taken for each batch state according to policy_net
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed
        # based on the "older" target_net; selecting their best reward with
        # max(1)[0]. This is merged based on the mask, such that we'll have
        # either the expected state value or 0 in case the state was final.
        next_q = self.target_net(next_state_batch).detach().max(1)[0]
        next_q_values = next_q * not_done_mask

        # Compute the expected Q values
        expected_q_values = (next_q_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values,
                                expected_q_values.unsqueeze(1))

        # Optimize the model
        self.opt.zero_grad()
        loss.backward()

        # Clip the gradients to lie between -1 and +1
        # for params in self.policy_net.parameters():
        #     params.grad.data.clamp_(-1, 1)

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


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def dqn(actions):
    return nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=8, stride=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, actions)
    )


def convert_state_to_tensor(state):
    state = np.ascontiguousarray(state, dtype=np.float32) / 255.
    state = torch.from_numpy(state)
    return state.unsqueeze(0)
