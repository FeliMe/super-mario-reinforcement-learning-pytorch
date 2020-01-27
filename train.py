import argparse
import gym_super_mario_bros
import numpy as np
import torch

from agent import DQNAgent, convert_state_to_tensor
from gym_super_mario_bros.actions import RIGHT_ONLY
from itertools import count
from nes_py.wrappers import JoypadSpace
from time import time
from wrappers import wrapper


def time_to_str(t):
    return "{:.0f}h {:.0f}m {:.0f}s".format(t // 3600, (t // 60) % 60, t % 60)


def time_left(t_start, n_iters, i_iter):
    iters_left = n_iters - i_iter
    time_per_iter = (time() - t_start) / i_iter
    time_remaining = time_per_iter * iters_left
    return time_to_str(time_remaining)


if __name__ == '__main__':

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_episodes', type=int, default=10000)
    parser.add_argument('-s', '--save_dir', type=str, default='models/')
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-p', '--model_path', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()

    if args.continue_training:
        assert args.model_path is not None

    # Build env (first level, right only)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrapper(env)

    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Agent
    agent = DQNAgent(actions=env.action_space.n,
                     max_memory=70000,
                     device=device,
                     save_dir=args.save_dir,
                     continue_training=args.continue_training,
                     model_path=args.model_path,
                     double_q=False)

    # Timing
    t_start = time()
    step = 0

    rewards = []

    for e in range(args.num_episodes):
        # Reset env
        state = env.reset()
        state = convert_state_to_tensor(state).to(device)

        # Reset total reward
        total_reward = 0

        # Play
        for t in count():

            # Show env
            if args.render:
                env.render()

            # Run agent
            action = agent.select_action(state=state)

            # Perform action
            next_state, reward, done, info = env.step(action=action.item())
            next_state = convert_state_to_tensor(next_state).to(device)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Replay
            agent.optimize_model()

            # Total reward
            total_reward += reward.item()
            if done or info['flag_get']:
                break

        # Rewards
        rewards.append(total_reward / t)

        if e % args.log_interval == 0:
            if device == 'cuda':
                max_memory = torch.cuda.max_memory_allocated(device) / 1e6
                print("Max memory on {}: {} MB\n".format(
                    device, int(max_memory)))
            time_elapsed = time() - t_start
            print('Episode {e} - '
                  'Frame (step) {f} - '
                  'Frames/sec {fs} - '
                  'Epsilon {eps} - '
                  'Mean Reward {r:.5f} - '
                  'Time elapsed {te} - '
                  'Time left {tl}'.format(
                      e=e,
                      f=agent.step,
                      fs=np.round((agent.step - step) / (time() - t_start)),
                      eps=np.round(agent.eps, 4),
                      r=np.mean(rewards[-100:]),
                      te=time_to_str(time_elapsed),
                      tl=time_left(t_start, args.num_episodes, e + 1)))

    print("Done")
    agent.save_model()
    env.close()
