from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch 
import torch.nn as nn 
import torch.optim as optim
import random
import numpy as np
from collections import deque
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
save_path = "DQN_hiv_model"
log_dir = "tensor"
# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!



class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(state))
        x = torch.relu(self.fc3(state))
        x = torch.relu(self.fc4(state))
        x = torch.relu(self.fc5(state))
        
        return self.fc6(x)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)  
    
class ProjectAgent:
    def __init__(self,model=None):
        self.state_dim = 6
        self.action_dim = 4
        self.gamma = 0.99
        self.batch_size = 64
        self.replay_buffer = ReplayBuffer(10000)
        
        
        self.q_network = DQNetwork(self.state_dim, self.action_dim)
        self.target_network = DQNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr =1e-3)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
    def act(self, observation, use_random=False, epsilon = 0.1):
        if use_random or random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            observation = torch.FloatTensor(observation)
            q_values = self.q_network(observation)
            return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        q_values = self.q_network(states).gather(1,actions)
        
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1,keepdim=True)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1-dones)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self):
        checkpoint = torch.load(save_path)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.q_network.eval()
        

 
           
def train_and_save_agent():
    replay_buffer = ReplayBuffer(10000)
    agent = ProjectAgent()
    num_episodes = 500
    target_update_frequency = 10
    for episode in range(num_episodes):
        print(f"Starting episode {episode+1}/{num_episodes}")
        observation,_ = env.reset()
        total_reward = 0
        for t in range(200):
            action = agent.act(observation)
            next_observation,reward , done , _, _ = env.step(action)
            agent.replay_buffer.push(observation,action,reward,next_observation,done)
            
            observation = next_observation
            total_reward += reward
            agent.train()
            if done:
                break
        if episode % target_update_frequency == 0:
            agent.update_target_network()
            print("Updated the Target Network.")
    agent.save(save_path)
    
    return None
    
    

if __name__== "__main__":
    train_and_save_agent()
    #agent = ProjectAgent()
    #agent.load()