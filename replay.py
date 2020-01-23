import argparse
import gym_super_mario_bros
import time

from agent import DQNAgent, convert_state_to_tensor
from itertools import count
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import wrapper


if __name__ == '__main__':

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('-n', '--n_replay', type=int, default=5)
    args = parser.parse_args()

    # Build env (first level, right only)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrapper(env)

    # Agent
    agent = DQNAgent(actions=env.action_space.n,
                     max_memory=100000,
                     device='cpu')

    for _ in range(args.n_replay):
        # Reset env
        state = env.reset()
        state = convert_state_to_tensor(state)

        total_reward = 0

        # Play
        for t in count():
            time.sleep(0.05)
            env.render()

            # Select action
            action = agent.policy_net(state).max(1)[1].view(1, 1)

            # Perform action
            next_state, reward, done, info = env.step(action=action.item())
            next_state = convert_state_to_tensor(next_state)
            total_reward += reward
            state = next_state

            if done:
                break
    env.close()
