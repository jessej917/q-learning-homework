import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

q_values = defaultdict(lambda: np.zeros(env.action_space.n))
epsilon = 0.5
learning_rate = 0.1
discount_factor = 0.9
training_error = []
n_episodes = 1000000

env = gym.make("Blackjack-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        # Action
        # print(f"\nYour hand: {obs[0]}, Dealer shows: {obs[1]}, Usable Ace: {obs[2]}")
        if np.random.random() < epsilon and episode < 8*n_episodes/10:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_values[obs]))
        
        next_obs, reward, done, _, info = env.step(action)

        # Print result
        # if reward == 1:
        #     print("\nYou won!")
        # elif reward == -1:
        #     print("\nYou lost!")
        # else:
        #     print("\nIt's a draw!")

        # Update
        # q_values[obs] += learning_rate * (reward + discount_factor*obs[0] - q_values[hand])
        future_q_value = (not done) * np.max(q_values[next_obs])
        temporal_difference = (reward + discount_factor * future_q_value - q_values[obs][action])

        q_values[obs][action] = q_values[obs][action] + learning_rate * temporal_difference
        training_error.append(temporal_difference)

        obs = next_obs

# Print result
# if reward == 1:
#     print("\nYou won!")
# elif reward == -1:
#     print("\nYou lost!")
# else:
#     print("\nIt's a draw!")

rolling_length = n_episodes//100
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
# axs[1].set_title("Episode lengths")
# length_moving_average = (
#     np.convolve(
#         np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
#     )
#     / rolling_length
# )
# axs[1].plot(range(len(length_moving_average)), length_moving_average)
# axs[2].set_title("Training Error")
# training_error_moving_average = (
#     np.convolve(np.array(training_error), np.ones(rolling_length), mode="same")
#     / rolling_length
# )
# axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()


env.close()
