import numpy as np
from collections import namedtuple, deque
from replay_buffer import ReplayBuffer

EPSILON = 1e-6

class PERBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, alpha, beta_start, beta_step, seed):
        super().__init__(action_size, buffer_size, batch_size, seed)
        self.priorities = deque(maxlen=buffer_size)
        self.idx_sampled_p = []
        self.p_max = 1
        self.alpha = alpha
        self.beta = beta_start
        self.beta_step = beta_step

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.p_max)

    def sample_list(self):
        self.idx_sampled_p = self.priority_sample()
        return [self.memory[i] for i in self.idx_sampled_p]

    def update_priorities(self, delta):
        for d, i in zip(delta, self.idx_sampled_p):
            self.priorities[i] = np.power(abs(d) + EPSILON, self.alpha)
        self.p_max = max(self.priorities)

    def compute_weights(self):
        N = len(self.priorities)
        total = sum(self.priorities)
        weights = [np.power(N*self.priorities[i]/total, -self.beta) for i in self.idx_sampled_p]
        self.beta = min(1.0, self.beta + self.beta_step)
        max_weight = max(weights)
        return [w/max_weight for w in weights]
    
    def priority_sample(self):
        root_node, leaf_nodes = create_tree(self.priorities)
        tree_total = root_node.value
        ranges = np.linspace(0.0, tree_total, num=self.batch_size+1)
        selected_idx = []
        for x, y in zip(ranges[:-1], ranges[1:]):
            rand_val = np.random.uniform(x, y)
            selected_idx.append(retrieve(rand_val, root_node).idx)    
        return selected_idx

# Sumtree implementation
class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self
    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf

def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    return nodes[0], leaf_nodes

def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node
    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)

def update(node: Node, new_value: float):
    change = new_value - node.value
    node.value = new_value
    propagate_changes(change, node.parent)

def propagate_changes(change: float, node: Node):
    node.value += change
    if node.parent is not None:
        propagate_changes(change, node.parent)