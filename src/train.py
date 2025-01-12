import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)
save_path = "PPO_hiv_model.pth"
# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.fc(state)

# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.fc(state)

class ProjectAgent:
    def __init__(self):
        self.state_dim = 6
        self.action_dim = 4
        self.gamma = 0.99
        self.lam = 0.95  
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.num_epochs = 10
        self.replay_buffer = []

        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim)
        self.value_network = ValueNetwork(self.state_dim)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=1e-3)

    def act(self, observation,use_random=False):
        if use_random:
            return np.random.randint(0,self.action_dim-1)
        state = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            
            action_probs = self.policy_network(state)
            
        action = np.random.choice(self.action_dim, p=action_probs.numpy()[0])
        return action

    def compute_advantages(self, rewards, dones, values, next_values):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def train(self):
        states, actions, rewards, dones, values, log_probs = zip(*self.replay_buffer)
        next_values = list(values[1:]) + [0]
        advantages, returns = self.compute_advantages(rewards, dones, values, next_values)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        old_log_probs = torch.FloatTensor(log_probs)

        for _ in range(self.num_epochs):
            
            action_probs = self.policy_network(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            
            ratios = torch.exp(new_log_probs - old_log_probs)

            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            
            value_preds = self.value_network(states).squeeze()
            value_loss = nn.MSELoss()(value_preds, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        self.replay_buffer = []

    def save(self, path):
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict()
        }, path)

    def load(self):
        checkpoint = torch.load("PPO_hiv_model.pth",map_location="cpu")
        
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.policy_network.eval()
        self.value_network.eval()
        print("Checkpoint loaded")
        

def train_and_save_agent():
    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
    agent = ProjectAgent()
    num_episodes = 500
    max_steps = 200

    for episode in range(num_episodes):
        print(f"Starting episode {episode+1}/{num_episodes}")
        observation, _ = env.reset()
        total_reward = 0
        trajectory = []

        for t in range(max_steps):
            action = agent.act(observation)
            next_observation, reward, done, _, _ = env.step(action)

            state_value = agent.value_network(torch.FloatTensor(observation).unsqueeze(0)).item()
            action_prob = agent.policy_network(torch.FloatTensor(observation).unsqueeze(0))[0][action].item()

            trajectory.append((
                observation, action, reward, done, state_value, np.log(action_prob)
            ))

            observation = next_observation
            total_reward += reward

            if done:
                break

        # Process trajectory
        agent.replay_buffer.extend(trajectory)
        agent.train()

    agent.save("PPO_hiv_model")

if __name__ == "__main__":
    train_and_save_agent()
