import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class DQN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 24)
#         self.fc2 = nn.Linear(24, 24)
#         self.fc3 = nn.Linear(24, output_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


env = gym.make("Acrobot-v1", render_mode="human")


output_size = env.action_space.n  # type: ignore
state, info = env.reset()
input_size = len(state)

policy_net = DQN(input_size, output_size)
state_dict = torch.load("acrobot_pytorch_dqn_policy.pth")


policy_net.load_state_dict(state_dict)
policy_net.eval()


def test_dqn_agent(episodes=10):
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        steps = 0

        while not done:
            env.render()
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action = policy_net(state_tensor).argmax().item()

                state, _, done, _, _ = env.step(action)
                steps += 1

        print(f"Test Episode {episode + 1}: Balanced for {steps} steps.")
    env.close()


test_dqn_agent()
