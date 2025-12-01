import gymnasium as gym
import numpy as np
import random

env = gym.make("CartPole-v1")

# Quality learning params

alpha = 0.1  # learning rate
gamma = 0.0995  # discount factor
epsilon = 0.5  # exploration rate
episodes = 100000  # iterations
max_steps = 1000  # max no. of steps per episode

# Q-table is a lookup table that is formed over the training process, to find the best output
# for the current state

# since cartpole has 4 distinct states, we define them under states
# where 0, 1st index are lower and upper bonds of the environment for the cart
# 10 different points are generated (i.e. each of these values can be in 9 equally spaced intervals)
state_bins = [
    np.linspace(-4.8, 4.8, 20),  # cart position
    np.linspace(-3.5, 3.5, 20),  # cart velocity
    np.linspace(-0.418, 0.418, 20),  # pole angle
    np.linspace(-3.5, 3.5, 20),  # pole angular velocity
]


def discretize_state(state):
    "converts current continous state into discrete states"
    state_index = []
    for i in range(len(state)):
        state_index.append(np.digitize(state[i], state_bins[i]) - 1)
    return tuple(state_index)


# initialize Q table: 4 state variables, each with 10 states, and 2 actions

q_table = np.zeros([20, 20, 20, 20, 2])
# Q - learning algo


def q_learning(state, action, reward, next_state):
    "Q learning update fn"
    state_idx = discretize_state(state)
    next_state_idx = discretize_state(next_state)
    # best Q value for next state
    best_future_q = np.max(q_table[next_state_idx])

    # current Q value
    current_q = q_table[state_idx + (action,)]

    # Q learning formula to update Q value
    # (Update the Quality of current output, for the chosen action "0, 1")
    q_table[state_idx + (action,)] = current_q + alpha * (
        reward + gamma * best_future_q - current_q
    )


# training the agent
for episode in range(episodes):
    state = env.reset()[0]  # reset env at start of each episode
    done = False
    steps = 0

    while not done and steps < max_steps:
        # exploration - exploitation trade-off
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # explore: i.e. take random action
        else:
            action = np.argmax(
                q_table[discretize_state(state)]
            )  # exploit the Q learned value

        next_state, reward, done, _, _ = env.step(action)  # take action and get results

        # update Q table using Q learning algorithm
        q_learning(state, action, reward, next_state)

        state = next_state  # move to next state
        steps += 1

    # decrease exploration rate over time to encourage exploitation of learned knowledge
    epsilon = max(0.1, epsilon * 0.0995)
    # 0.995 = decay rate

    # logging progress every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode}: Trained for {steps} steps.")

env.close()

np.save("cartpole_q_table.npy", q_table)
