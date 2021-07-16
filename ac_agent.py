import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int): number of agents
            config (dict): hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(config.seed)
        self.config = config

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, config.seed, config.first_layer_size, config.second_layer_size, \
                                 config.dropout_rate, config.bn_active).to(device)
        self.actor_target = Actor(state_size, action_size, config.seed, config.first_layer_size, config.second_layer_size, \
                                  config.dropout_rate, config.bn_active).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(n_agents*state_size, n_agents*action_size, config.seed, config.first_layer_size,\
                                   config.second_layer_size, config.dropout_rate, config.bn_active).to(device)
        self.critic_target = Critic(n_agents*state_size, n_agents*action_size, config.seed, config.first_layer_size,\
                                    config.second_layer_size, config.dropout_rate, config.bn_active).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic)

        # Noise process
        self.noise = OUNoise(action_size, config.seed, sigma=config.sigma)
        self.epsilon = config.epsilon_start


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.epsilon = max(self.epsilon * self.config.epsilon_decay, self.config.epsilon_min)
            action += self.noise.sample() * self.epsilon
        return np.clip(action, -1, 1)
    
    def act_target(self, state):
        """Returns actions for given state as per current policy."""
        return self.actor_target(state)

    def reset(self):
        self.noise.reset()
        
    def save(self, prefix='checkpoint', suffix=''):
        torch.save(self.actor_local.state_dict(), self.config.path+prefix+'_actor'+suffix+'.pth')
        torch.save(self.critic_local.state_dict(), self.config.path+prefix+'_critic'+suffix+'.pth')


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

