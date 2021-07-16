import torch
import torch.nn.functional as F
from ac_agent import Agent
from replay_buffer import ReplayBuffer, deque, namedtuple

class DDPG(Agent):
    def __init__(self, state_size, action_size, config):
        self.n_agents = 1
        super().__init__(state_size, action_size, self.n_agents, config)
        # Replay memory
        self.memory = self.init_replay_buffer()
        # learn counter
        self.postpone_learning = 0
        self.alg = 'ddpg'
                      
    def init_replay_buffer(self):
        return ReplayBuffer(self.action_size, self.config.buffer_size, self.config.batch_size, self.config.seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        [self.memory.add(state, action, reward, next_state, done) for state, action, reward, next_state, done\
         in zip(states, actions, rewards, next_states, dones)]

        # Learn, if enough samples are available in memory
        self.postpone_learning = (self.postpone_learning + 1) % self.config.learn_every
        if len(self.memory) > self.config.batch_size and not self.postpone_learning:
            experiences = self.memory.sample()
            self.learn(experiences, self.config.gamma)

    def compute_critic_loss(self, Q_expected, Q_target):
        return F.mse_loss(Q_expected, Q_target)

    def learn(self, experiences, gamma):
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

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = self.compute_critic_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.limit_gradients:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)
        return Q_targets, Q_expected

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