import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

# Define a simple neural network for policy and value estimation
class PolicyValueNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(24, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to select actions based on policy
# def select_action(policy_network, state):
#     with torch.no_grad():
#         logits = policy_network(torch.tensor(state).float())
#         action = torch.argmax(logits).item()
#     return action

def select_action(policy_network, state, episodes):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episodes / EPS_DECAY)
    if sample > eps_threshold or episodes == -1:
        with torch.no_grad():
            logits = policy_network(torch.tensor(state).float())
            return torch.argmax(logits).item()
    else:
        return random.randrange(0,4)

# Function to train the policy network using REINFORCE algorithm
def train_policy(policy_network, optimizer, rewards, log_probs):
    policy_loss = []
    for reward, log_prob in zip(rewards, log_probs):
        policy_loss.append(-log_prob * reward)
    
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

# Define a custom environment
class CustomEnvironment:
    def __init__(self):
        self.observation_space = 9
        self.action_space = 4
        self.rounds = 0
        self.initialmap = [0,0,0,0,0,0,0,0,0]
        self.startloc = [0,2]
        self.loc = self.startloc.copy()
        self.state = self.initialmap.copy()
    
    def reset(self):
        self.loc = self.startloc.copy()
        self.state = self.game_generate_new_state()
        self.done = False
        self.rounds = 0
        return self.state

    def game_generate_new_state(self):
        newstate = self.initialmap.copy()
        for x in range(3):
            for y in range(3):
                if x == self.loc[0] and y == self.loc[1]:
                    newstate[y*3+x] = 1
        return newstate
    
    def step(self, action):
        dx = 0
        dy = 0
        if action == 0:
            dy = -1
        elif action == 1:
            dy = 1
        elif action == 2:
            dx = -1
        elif action == 3:
            dx = 1

        newx = self.loc[0] + dx
        newy = self.loc[1] + dy

        reward = 0

        # hitting wall
        if newx < 0 or newy < 0 or newx > 2 or newy > 2:
          dx = 0
          dy = 0
          reward -= 1

        newx = self.loc[0] + dx
        newy = self.loc[1] + dy

        self.loc = [newx, newy]
        self.state = self.game_generate_new_state()
        done = False

        self.rounds += 1

        if self.loc[0] == 2 and self.loc[1] == 0:
            done = True
            reward = 1
        elif self.rounds > 10:
            # time limit
            done = True
        return self.state, reward, done, {}
    
    def calculate_reward(self, action):
        return 1

# Initialize the custom environment
env = CustomEnvironment()

# Initialize the policy network
policy_network = PolicyValueNetwork(env.observation_space, env.action_space)
optimizer = optim.AdamW(policy_network.parameters(), lr=0.01)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    rewards = []
    log_probs = []
    done = False
    
    while not done:
        action = select_action(policy_network, state, episode)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        
        # Store rewards and log probabilities
        rewards.append(reward)
        log_probs.append(torch.log_softmax(policy_network(torch.tensor(state).float()), dim=0)[action])
        
        if done:
            # Calculate discounted rewards
            discounted_rewards = []
            running_reward = 0
            for r in reversed(rewards):
                running_reward = r + 0.99 * running_reward
                discounted_rewards.insert(0, running_reward)
            
            # Normalize discounted rewards
            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
            
            # Train policy network
            train_policy(policy_network, optimizer, discounted_rewards, log_probs)
            
            if episode % 10 == 0:
                print("Episode {}: Total Reward = {}".format(episode, sum(rewards)))
            break

# Testing the trained policy
state = env.reset()
done = False
print(env.loc)
while not done:
    action = select_action(policy_network, state, -1)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    print("Action:", action, "Reward:", reward)
    print(env.loc)
