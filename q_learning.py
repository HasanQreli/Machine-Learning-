import numpy as np
from enum import Enum

# using a 2D array as the map
# 1's are rewards, 0's no effect, 2 is the end
gridworld = np.array([
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0,  -1,   0,  0, -1, 0],
    [0,   0,   0,  0,  0, 0,   0,   0,  0,  0, 0,   0,   0,  0,  0, 1]

])

n_observations = gridworld.shape[0]*gridworld.shape[1]
n_actions = 4

# Initialize the Q-table to 0
Q_table = np.zeros((n_observations, n_actions))
n_episodes = 10000
max_iter_episode = 1000
# parameters used for exploring randomly or based on previous knowledge
exploration_proba = 1
exploration_decreasing_decay = 0.001
min_exploration_proba = 0.01

gamma = 0.99
# learning rate
lr = 0.1
rewards_per_episode = list()


class Action(Enum):
    left = 0
    right = 1
    up = 2
    down = 3

# function to move on gridworld
# 0: <-, 1: ->, 2: ^, 3: v
def move(action, current_state):
    if action == Action.left.value and current_state[1] != 0:
        return (current_state[0], current_state[1]-1)
    if action == Action.right.value and current_state[1] != gridworld.shape[1]-1:
        return (current_state[0], current_state[1]+1)
    if action == Action.up.value and current_state[0] != 0:
        return (current_state[0]-1, current_state[1])
    if action == Action.down.value and current_state[0] != gridworld.shape[0]-1:
        return (current_state[0]+1, current_state[1])
    else:
        return current_state


# we iterate over episodes
for e in range(n_episodes):
    # we initialize the first state of the episode which is always the top left state
    # 2 element tuple represent coordinates
    current_state = (0, 0)
    done = False

    # sum the rewards that the agent gets from the environment
    total_episode_reward = 0

    for i in range(max_iter_episode):
        # choosing to go random or based on previous info
        # randomly choosing
        if np.random.uniform(0, 1) < exploration_proba:
            action = np.random.randint(4)
        # choosing based on our previous observations
        else:
            action = np.argmax(Q_table[current_state[0] * gridworld.shape[0] + current_state[1], :])


        # The environment runs the chosen action and returns
        # the next state, a reward and true if the episode is ended.
        next_state = move(action, current_state)
        reward = gridworld[next_state[0], next_state[1]]
        done = True if (gridworld[next_state[0], next_state[1]] == 1) else False

        # We update our Q-table using the Q-learning iteration
        # convert index into a unique integer
        cs = current_state[0] * gridworld.shape[0] + current_state[1]
        ns = next_state[0] * gridworld.shape[0] + next_state[1]
        Q_table[cs, action] = (1 - lr) * Q_table[cs, action] + lr * (
                    reward + gamma * max(Q_table[ns, :]))

        # update reward
        total_episode_reward = total_episode_reward + reward

        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state

    # We update the exploration proba using exponential decay formula
    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))
    rewards_per_episode.append(total_episode_reward)


action_arr = ["<", ">", "^", "v"]
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if(gridworld[i, j] == -1):
            print("@", end=" ")
        elif(gridworld[i, j] == 1):
            print("S", end=" ")
        else:
            print(action_arr[np.argmax(Q_table[i*gridworld.shape[0] + j, :])], end=" ")
    print()



# checking how the reward changes as we learn more
print("Mean reward per thousand episodes")
for i in range(10):
    print((i+1)*1000, ": mean episode reward: ", np.mean(rewards_per_episode[1000*i:1000*(i+1)]))