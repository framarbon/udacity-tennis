import torch
import torch.nn.functional as F
from ac_agent import Agent
from replay_buffer import ReplayBuffer, deque, namedtuple
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, state_size, action_size, config):

        self.alg = 'maddpg'
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = 2
        self.config = config
        self.agents = [Agent(state_size, action_size, self.n_agents, config) for i in range(config.n_agents)]
        # Replay memory
        self.memory = ReplayBuffer(action_size, config.buffer_size, config.batch_size, config.seed)
        
        # learn counter
        self.postpone_learning = 0
        
    def act(self, state_all_agents, add_noise=True):
        return [agent.act(obs, add_noise) for agent, obs in zip(self.agents, state_all_agents)]
    
    def act_targets(self, state_all_agents):
        return torch.cat([agent.act_target(obs).unsqueeze(1) for agent, obs in zip(self.agents, state_all_agents)], 1)
    
    def unsqueeze(self, state, action, reward, next_state, done):
        return np.asarray(state)[np.newaxis], \
                np.asarray(action)[np.newaxis], \
                np.asarray(reward)[np.newaxis], \
                np.asarray(next_state)[np.newaxis], \
                np.asarray(done)[np.newaxis]
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(*self.unsqueeze(state, action, reward, next_state, done))
        
        # Learn, if enough samples are available in memory
        self.postpone_learning = (self.postpone_learning + 1) % self.config.learn_every
        if len(self.memory) > self.config.batch_size and not self.postpone_learning:
            for i in range(self.n_agents):
                experiences = self.memory.sample()
                self.learn(experiences, i, self.config.gamma)
                
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    def learn(self, experiences, agent_i, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        agent = self.agents[agent_i]
        
        states_flat = states.view(self.config.batch_size, -1)
        states_flat_next = next_states.view(self.config.batch_size, -1)
        
        states = states.transpose(0,1)
        next_states = next_states.transpose(0,1)
        rewards = rewards.transpose(0,1)
        dones = dones.transpose(0,1)

        index = torch.tensor([agent_i]).to(device)

        state_agent = states.index_select(dim=1, index=index)
        next_state_agent = next_states.index_select(dim=1, index=index)
        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.act_targets(next_states)
        
        with torch.no_grad():
            Q_targets_next = agent.critic_target(states_flat_next, actions_next.view(self.config.batch_size, -1)).detach().squeeze()
        
        # Compute Q targets for current states (y_i)
        Q_targets = (rewards[agent_i] + (gamma * Q_targets_next * (1 - dones[agent_i]))).unsqueeze(1).detach()
        # Compute critic loss
        Q_expected = agent.critic_local(states_flat, actions.view(self.config.batch_size, -1))
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
#         critic_loss = F.smooth_l1_loss(Q_expected, Q_targets.detach())

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.limit_gradients:
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        q_input = [ self.agents[i].actor_local(ob) if i == agent_i\
                   else self.agents[i].actor_local(ob).detach()
                   for i, ob in enumerate(states) ]
        q_input = torch.cat(q_input, dim=1)

#         actions_pred = self.actor_local[agent_i](state_agent)
        actor_loss = -agent.critic_local(states_flat, q_input).mean()
    
    
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(agent.critic_local, agent.critic_target, self.config.tau)
        self.soft_update(agent.actor_local, agent.actor_target, self.config.tau)       
        return Q_targets, Q_expected

    def save(self, prefix='checkpoint'):
        [agent.save(prefix=prefix, suffix='_'+str(i)) for i, agent in enumerate(self.agents)]
