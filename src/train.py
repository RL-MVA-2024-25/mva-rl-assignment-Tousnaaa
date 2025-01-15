import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from collections import deque
from evaluate import evaluate_HIV
import random
from pathlib import Path
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)
#Hardcoded path of the model
save_path = "DQN_hiv_model"

class DQNetwork(nn.Module):
    """
    Model for the q_values prediction.
    """
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, state):
        return self.fc(state)
class ReplayBuffer:
    """
    Replay buffer class for the training.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)
class ProjectAgent:
    def __init__(self):
        
        #Exploration term -> 1 at the beginning of the training
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        
        
        self.state_dim = 6
        self.action_dim = 4
        self.gamma = 0.95
        
        self.batch_size = 256
        self.replay_buffer = ReplayBuffer(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #q_network and target network initialization
        self.q_network = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def act(self, observation,use_random=False):
        if use_random and random.random() < self.epsilon:
            #We sample a random element from the action space 
            return env.action_space.sample()
        else:
            #Or we take the element with the highest estimated Q
            observation = torch.tensor(observation,dtype = torch.float32,device=self.device).unsqueeze(0)
            q_values = self.q_network(observation) 
            return torch.argmax(q_values).item()
                


    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states,dtype = torch.float32,device=self.device)
        actions = torch.tensor(actions,dtype = torch.long,device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards,dtype=torch.float32,device = self.device).unsqueeze(1)
        next_states = torch.tensor(next_states,dtype=torch.float32,device = self.device)
        dones = torch.tensor(dones,dtype=torch.float32,device = self.device).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
    
        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def load(self):
        checkpoint = torch.load(save_path,map_location = self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.q_network.eval()
        

def train_and_save_agent():
    agent = ProjectAgent()
    num_episodes = 1000
    target_update_frequency = 400
    max_episode_reward = 0
    id_ep_max = 0
    for episode in tqdm(range(num_episodes)):
        print(f"Starting episode {episode+1}/{num_episodes}")
        observation, _ = env.reset()
        total_reward = 0
        done = False
        trunc = False
        while not (done or trunc):
            action = agent.act(observation,use_random=True)
            next_observation, reward, done, trunc, _ = env.step(action)

            agent.replay_buffer.push(observation, action, reward, next_observation, done)

            observation = next_observation
            total_reward += reward

            agent.train()

        agent.epsilon = max(agent.epsilon_min,agent.epsilon*agent.epsilon_decay)
      
        if episode % target_update_frequency == 0:
            agent.update_target_network()
            print("Updated target network.")
        print(f"Reward : {total_reward}, Epsilon : {agent.epsilon}")  
        if total_reward > max_episode_reward:
            max_episode_reward = total_reward
            id_ep_max = episode
            
            print(f"New max reward.")
            agent.save(f"checkpoints/model_rew_{total_reward}_{episode}")
    print(f"Best episode reward was reached on episode {id_ep_max}")
    agent.save("DQN_hiv_model")
    
    

if __name__ == "__main__":
    #Launch training
    train_and_save_agent()
