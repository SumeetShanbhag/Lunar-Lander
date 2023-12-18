import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import gym
import gym.spaces as sp
from tqdm import trange
from time import sleep
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import os
import imageio
from PIL import Image, ImageDraw

# Setting up the device for PyTorch (will use CUDA if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definition of the Q-Network, used by the DQN agent
class QNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=128):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(n_states, n_hidden) # First fully connected layer
        self.layer2 = nn.Linear(n_hidden, n_hidden) # Second fully connected layer
        self.output_layer = nn.Linear(n_hidden, n_actions) # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.layer1(x)) # Activation function applied to the first layer
        x = F.relu(self.layer2(x)) # Activation function applied to the second layer
        return self.output_layer(x) # Return the output

# Deep Q-Network (DQN) Agent  
class DQN():
    def __init__(self, n_states, n_actions, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=5, tau=1e-3):
        self.n_actions = n_actions
        self.net_eval = QNet(n_states, n_actions).to(device)  # Evaluation network
        self.net_target = QNet(n_states, n_actions).to(device)  # Target network for stable learning
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=lr)  # Optimizer
        self.criterion = nn.MSELoss()  # Loss function
        self.memory = ReplayBuffer(n_actions, mem_size, batch_size)  # Replay memory buffer
        self.counter = 0  # Counter for updating the target network
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor for future rewards
        self.learn_step = learn_step  # Frequency of learning/updating
        self.tau = tau  # Rate of updating target network

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.net_eval.eval()
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()

        if random.random() < epsilon:
            return random.choice(range(self.n_actions))
        else:
            return torch.argmax(action_values).item()

    def save2memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_target = self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)          # target, if terminal then y_j = rewards
        q_eval = self.net_eval(states).gather(1, actions)

        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.softUpdate()

    def softUpdate(self):
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)

# Replay Buffer for storing experiences
class ReplayBuffer():
    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = deque(maxlen = memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

# Training function for the DQN agent    
def train(env, agent, n_episodes=2000, max_steps=1000, eps_start=1.0, eps_end=0.1, eps_decay=0.995, target=200, chkpt=False):
    score_hist = []
    epsilon = eps_start

    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    # bar_format = '{l_bar}{bar:10}{r_bar}'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)
    for idx_epi in pbar:
        state = env.reset()
        score = 0
        for idx_step in range(max_steps):
            action = agent.getAction(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.save2memory(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        score_hist.append(score)
        score_avg = np.mean(score_hist[-100:])
        epsilon = max(eps_end, epsilon*eps_decay)

        pbar.set_postfix_str(f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}")
        pbar.update(0)

        # Early stop
        if len(score_hist) >= 100:
            if score_avg >= target:
                break

    if (idx_epi+1) < n_episodes:
        print("\nTarget score reached! Stopping training.")
    else:
        print("\Training complete!")
        
    if chkpt:
        torch.save(agent.net_eval.state_dict(), 'checkpoint.pth')

    return score_hist

# Function to test the trained agent in the Lunar Lander environment
def testLander(env, agent, loop=3):
    for i in range(loop):
        state = env.reset()
        for idx_step in range(500):
            action = agent.getAction(state, epsilon=0)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break
    env.close()

# Function to plot the training scores    
def plotScore(scores):
    plt.figure()
    plt.plot(scores)
    plt.title("Score History")
    plt.xlabel("Episodes")
    plt.show()

# Setting parameters for training
BATCH_SIZE = 128       # minibatch size
LR = 1e-3               # learning rate
EPISODES = 5000         # max number of training episodes
TARGET_SCORE = 250     # early training stop at avg score of last 100 episodes
GAMMA = 0.99            # discount factor
MEMORY_SIZE = 10000     # max memory buffer size
LEARN_STEP = 5          # how often to learn
TAU = 1e-3              # for soft update of target parameters
SAVE_CHKPT = True      # save trained network .pth file

# Initialize the Lunar Lander environment
env = gym.make('LunarLander-v2')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the DQN agent
agent = DQN(n_states = num_states, n_actions = num_actions, batch_size = BATCH_SIZE,
    lr = LR, gamma = GAMMA, mem_size = MEMORY_SIZE, learn_step = LEARN_STEP, tau = TAU)

# Train the agent
score_hist = train(env, agent, n_episodes=EPISODES, target=TARGET_SCORE, chkpt=SAVE_CHKPT)
plotScore(score_hist) # Plot the scores after training

# Clear CUDA cache if CUDA is used
if str(device) == "cuda":
    torch.cuda.empty_cache()

# Test the trained agent
testLander(env, agent, loop=10)

# Function to overlay text on images
def TextOnImg(img, score):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"Score={score: .2f}", fill=(255, 255, 255))

    return np.array(img)

# Function to save a series of frames as a GIF
def save_frames_as_gif(frames, filename, path="gifs/"):
    if not os.path.exists(path):
        os.makedirs(path)
        
    print("Saving gif...", end="")
    imageio.mimsave(path + filename + ".gif", frames, fps=60)

    print("Done!")

# Function to record gameplay and save as GIF
def gym2gif(env, agent, filename="gym_animation", loop=3):
    frames = []
    for i in range(loop):
        state = env.reset()
        score = 0
        for idx_step in range(500):
            frame = env.render(mode="rgb_array")
            frames.append(TextOnImg(frame, score))
            action = agent.getAction(state, epsilon=0)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
    env.close()
    save_frames_as_gif(frames, filename=filename)

# Record and save gameplay as GIF
gym2gif(env, agent, loop=5)