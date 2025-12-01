import gymnasium as gym
import numpy as np

# q_table = np.load("cartpole_q_table.npy")

q_table = np.zeros([20, 20, 20, 20, 2])

env = gym.make("CartPole-v1", render_mode="human")


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


def test_agent(episodes=1):
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        steps = 0

        while not done:
            env.render()
            action = np.argmax(q_table[discretize_state(state)])
            state, _, done, _, _ = env.step(action)
            steps += 1

        print(f"Episode {episode + 1}: Agent balanced for {steps} steps.")

    env.close()


test_agent(episodes=1)
