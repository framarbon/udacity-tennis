from ddpg_agent import DDPG
from per_buffer import PERBuffer

import numpy as np

class PDDPG(DDPG):
    def __init__(self, state_size, action_size, config):
        super().__init__(state_size, action_size, config)
        self.alg = 'pddpg'
           
    def init_replay_buffer(self):
        print('Setting up Prioritized Experience Replay...')
        return PERBuffer(self.action_size, self.config.buffer_size, self.config.batch_size, self.config.alpha, self.config.beta, self.config.beta_step, self.config.seed)

    def compute_loss(self, Q_expected, Q_target):
        loss = F.mse_loss(Q_expected, Q_target, reduce=False)
        weights = torch.FloatTensor(self.memory.compute_weights()).to(device)
        return torch.mean(weights * loss)

    def learn(self, experiences, gamma):
        Q_target, Q_expected =super().learn(experiences, gamma)
        self.memory.update_priorities((Q_target - Q_expected).cpu().data.numpy())