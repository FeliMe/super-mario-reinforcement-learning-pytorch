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
                     max_memory=100000,
                     device=device,
                     save_dir=args.save_dir,
                     continue_training=args.continue_training,
                     model_path=args.model_path)

    # Timing
    start = time()
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

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Replay
            agent.optimize_model()

            # Total reward
            total_reward += reward.item()
            if done:
                break

        # Rewards
        rewards.append(total_reward / t)

        if e % args.log_interval == 0:
            print('Episode {e} - '
                  'Frame (step) {f} - '
                  'Frames/sec {fs} - '
                  'Epsilon {eps} - '
                  'Mean Reward {r}'.format(
                      e=e,
                      f=agent.step,
                      fs=np.round((agent.step - step) / (time() - start)),
                      eps=np.round(agent.eps, 4),
                      r=np.mean(rewards[-100:])))

    print("Done")
    agent.save_model()
    env.close()
