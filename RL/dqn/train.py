import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt  # 1. Added for plotting

# --- Environment & Network Setup ---
env = gym.make("CartPole-v1", render_mode=None)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


input_size = 4
output_size = 2
device = torch.device("cpu")

policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
replay_buffer = deque(maxlen=20000)


# --- Helper Functions ---
def store_experience(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))


def sample_experience(batch_size):
    return random.sample(replay_buffer, batch_size)


# --- Hyperparameters ---
batch_size = 64 * 4
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
target_update_frequency = 100
episodes = 20000
max_steps_per_episode = 500

# --- 2. Tracking Lists ---
reward_history = []
epsilon_history = []
loss_history = []
avg_q_history = []

print("Starting Training...")

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    # Temporary lists to track metrics within THIS episode
    episode_losses = []
    episode_q_values = []

    for t in range(max_steps_per_episode):
        # --- Action Selection ---
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                action = policy_net(state_tensor.unsqueeze(0)).argmax().item()

        # --- Step ---
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward  # type: ignore

        store_experience(state, action, reward, next_state, done)
        state = next_state

        # --- Training Step ---
        if len(replay_buffer) > batch_size and t % 4 == 0:
            experiences = sample_experience(batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.as_tensor(np.array(states), dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.long)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Current Q Values
            current_q_values = (
                policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            )

            # Next Q Values (Target)
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = loss_fn(current_q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- 3. Capture Diagnostics ---
            episode_losses.append(loss.item())
            # Track the average Q-value of the batch to see if values rise
            episode_q_values.append(current_q_values.mean().item())

        if done:
            break

    # --- End of Episode Logic ---
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % target_update_frequency == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Update Histories
    reward_history.append(total_reward)
    epsilon_history.append(epsilon)

    # Calculate average loss/Q for the episode (if training happened)
    avg_loss = np.mean(episode_losses) if episode_losses else 0
    avg_q = np.mean(episode_q_values) if episode_q_values else 0

    loss_history.append(avg_loss)
    avg_q_history.append(avg_q)

    if episode % 50 == 0:
        print(
            f"Episode {episode}: Reward: {total_reward}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}"
        )

env.close()

# --- 4. Plotting Trends ---
print("Training Complete. Generating Plots...")

# Create a 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Rewards
axs[0, 0].plot(reward_history, label="Total Reward", color="blue", alpha=0.6)
# Add a moving average trend line (window 50)
if len(reward_history) > 50:
    moving_avg = np.convolve(reward_history, np.ones(50) / 50, mode="valid")
    axs[0, 0].plot(
        range(49, len(reward_history)),
        moving_avg,
        color="red",
        label="50-Ep Moving Avg",
    )
axs[0, 0].set_title("Reward per Episode")
axs[0, 0].set_xlabel("Episode")
axs[0, 0].set_ylabel("Total Reward")
axs[0, 0].legend()

# Plot 2: Loss
axs[0, 1].plot(loss_history, label="Avg Loss", color="orange")
axs[0, 1].set_title("Average Training Loss per Episode")
axs[0, 1].set_xlabel("Episode")
axs[0, 1].set_ylabel("MSE Loss")

# Plot 3: Epsilon Decay
axs[1, 0].plot(epsilon_history, label="Epsilon", color="green")
axs[1, 0].set_title("Epsilon Decay")
axs[1, 0].set_xlabel("Episode")
axs[1, 0].set_ylabel("Epsilon")

# Plot 4: Average Q-Value
axs[1, 1].plot(avg_q_history, label="Avg Q-Value", color="purple")
axs[1, 1].set_title("Average Q-Value Estimation")
axs[1, 1].set_xlabel("Episode")
axs[1, 1].set_ylabel("Q-Value")

plt.tight_layout()

plt.savefig("training_trends.png", dpi=300)  # Save as PNG with high resolution
print("Training trends saved to 'training_trends.png'")

plt.show()

# Save Model
torch.save(policy_net.state_dict(), "cartpole_new_dqn_policy.pth")
print("Policy Network saved successfully.")
