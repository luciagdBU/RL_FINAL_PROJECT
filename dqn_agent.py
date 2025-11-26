# dqn_agent.py  –  network, replay buffer, ε‑greedy, optimiser
from __future__ import annotations
import math, random
from collections import deque
from typing import Tuple, List
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────── network ────────────────
class DQN(nn.Module):
    """
    TODO: IMPLEMENT THIS CLASS
    
    Deep Q-Network using Convolutional Neural Networks for Pac-Man.
    
    CONCEPT:
    The DQN approximates the Q-function Q(state, action) using a neural network.
    For Pac-Man, the state is an RGB image of the game board, so we use CNNs
    to extract spatial features, then fully connected layers to output Q-values.
    
    ARCHITECTURE DESIGN:
    Your network should process game screenshots (RGB images) and output 
    Q-values for each possible action. Consider:
    - Convolutional layers to detect game features (walls, pellets, ghosts)
    - Pooling or strided convolutions to reduce spatial dimensions
    - Fully connected layers to combine features and output action values
    - Appropriate activation functions (ReLU is common)
    
    PARAMETERS for __init__:
    - obs_shape: Tuple[int, int, int] - Shape of input observations (H, W, C)
    - n_actions: int - Number of possible actions (4 for Pac-Man: up/down/left/right)
    
    METHODS TO IMPLEMENT:
    - __init__(self, obs_shape, n_actions): Initialize layers
    - forward(self, x): Forward pass through network
    
    INPUT FORMAT:
    - x arrives as (batch_size, H, W, C) uint8 tensor from environment
    - You need to convert to float, normalize to [0,1], and rearrange to (B, C, H, W)
    - PyTorch conv layers expect channels-first format: (batch, channels, height, width)
    
    OUTPUT FORMAT:
    - Return tensor of shape (batch_size, n_actions) with Q-values for each action
    
    
    TORCH FUNCTIONS YOU'LL NEED:
    - nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    - nn.ReLU()
    - nn.Flatten()
    - nn.Linear(in_features, out_features)
    - nn.Sequential() to chain layers
    - tensor.float() to convert to float
    - tensor.permute(0, 3, 1, 2) to rearrange dimensions
    - torch.zeros() for dummy input to calculate dimensions
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        # TODO: Implement network architecture
        H, W, C = obs_shape
        self.n_actions = n_actions # store number of actions for later use

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            conv_out_size = self.conv(dummy_input).reshape(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    
    def forward(self, x: torch.Tensor):
        """
        TODO: IMPLEMENT FORWARD PASS
        
        Process input through your network architecture.
        
        STEPS:
        1. Convert uint8 input to float and normalize by dividing by 255.0
        2. Rearrange from (B,H,W,C) to (B,C,H,W) using permute
        3. Pass through convolutional layers
        4. Pass through fully connected layers
        5. Return Q-values for each action
        """
        # TODO: Implement forward pass
        # (B, H, W, C) → (B, C, H, W)
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)

        # CNN feature extraction
        features = self.conv(x)

        # Flatten and get Q-values
        features = features.reshape(features.size(0), -1)
        q_values = self.fc(features)
        return q_values
    

# ───────────── replay buffer ─────────────
class ReplayMemory:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(tuple(transition))

    def sample(self, batch_size: int):
        s = random.sample(self.buf, batch_size)
        return map(np.array, zip(*s))

    def __len__(self):
        return len(self.buf)

# ───────────── ε‑greedy & optimise ───────
def select_action(state: np.ndarray, net: DQN, step: int,
                  eps_start: float, eps_end: float, eps_decay: int) -> int:
    """
    TODO: IMPLEMENT THIS FUNCTION
    
    Select an action using epsilon-greedy exploration strategy.
    
    CONCEPT:
    Epsilon-greedy balances exploration vs exploitation:
    - With probability epsilon: choose random action (exploration)
    - With probability (1-epsilon): choose action with highest Q-value (exploitation)
    - Epsilon should decay over time (start high for exploration, end low for exploitation)
    
    PARAMETERS:
    - state: np.ndarray - Current game state (image)
    - net: DQN - The neural network that estimates Q-values
    - step: int - Current training step (used for epsilon decay)
    - eps_start: float - Starting epsilon value (e.g., 1.0 = 100% random)
    - eps_end: float - Final epsilon value (e.g., 0.05 = 5% random)
    - eps_decay: int - How quickly epsilon decays (larger = slower decay)
    
    EPSILON DECAY:
    Use exponential decay: eps = eps_end + (eps_start - eps_end) * exp(-step / eps_decay)
    This gives you a smooth transition from high exploration to low exploration.
    
    IMPLEMENTATION STEPS:
    1. Calculate current epsilon using decay formula
    2. Generate random number and compare with epsilon
    3. If random < epsilon: return random action
    4. Else: use network to get Q-values and return best action
    
    TORCH FUNCTIONS YOU'LL NEED:
    - torch.as_tensor() to convert numpy to torch tensor
    - tensor.unsqueeze(0) to add batch dimension
    - torch.no_grad() context manager for inference
    - tensor.argmax() to find action with highest Q-value
    - int() to convert tensor to Python int
    
    RANDOM FUNCTIONS:
    - random.random() returns float in [0,1)
    - random.randrange(n) returns int in [0, n)
    - math.exp() for exponential function
    
    DEVICE HANDLING:
    - Move tensors to DEVICE before passing to network
    - Network is already on DEVICE, so inputs must match
    
    RETURN TYPE: int
    - Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
    """
    
    # TODO: Implement epsilon-greedy action selection
    # Compute current epsilon (exploration rate) with exponential decay
    eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)

    # Number of actions from the network (do not hardcode 4)
    n_actions = net.n_actions

    # Exploration vs exploitation
    if random.random() < eps:
        # Exploration: choose a random action
        action = random.randrange(n_actions)
    else:
        # Exploitation: choose the best action according to the network
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = net(state_t)            # shape: (1, n_actions)
            action = int(q_values.argmax(dim=1).item())

    return action

def optimise(memory: ReplayMemory, policy: DQN, target: DQN,
             optimiser: optim.Optimizer, batch_size: int, gamma: float):
    """
    TODO: IMPLEMENT THIS FUNCTION
    
    Perform one step of DQN optimization using experience replay and target network.
    This is the CORE of the DQN algorithm!
    
    CONCEPT:
    DQN uses two key innovations:
    1. Experience Replay: Store transitions and learn from random batches
    2. Target Network: Use separate network for computing targets (more stable)

    Recall discussion in class about the DQN update rule, and how we can write it as minimizing a loss function.
    
    PARAMETERS:
    - memory: ReplayMemory - Buffer containing stored experiences
    - policy: DQN - Main network being trained
    - target: DQN - Target network (updated less frequently)  
    - optimiser: torch optimizer - Updates policy network weights
    - batch_size: int - Number of experiences to sample
    - gamma: float - Discount factor for future rewards
    
    IMPLEMENTATION STEPS:
    1. Check if memory has enough experiences (return early if not)
    2. Sample batch of experiences: (state, action, reward, next_state, done)
    3. Convert numpy arrays to torch tensors on correct device
    4. Compute Q-values for current states using policy network
    5. Compute target Q-values for next states using target network
    6. Compute target values: reward + gamma * max_next_Q * (1 - done)
    7. Compute loss between current Q-values and targets
    8. Perform gradient descent step
    
    EXPERIENCE REPLAY SAMPLING:
    - memory.sample(batch_size) returns: states, actions, rewards, next_states, dones
    - Each is a numpy array with batch_size elements
    - Convert all to torch tensors on DEVICE
    
    COMPUTING CURRENT Q-VALUES:
    - Use policy(states) to get Q-values for all actions
    
    COMPUTING TARGET Q-VALUES:
    - Use torch.no_grad() to prevent gradients flowing to target network
    - Use target(next_states) to get Q-values for next states
    - Compute: rewards + (1 - dones) * gamma * max_next_q_values
    
    LOSS AND OPTIMIZATION:
    - Call optimiser.zero_grad() to clear old gradients
    - Call loss.backward() to compute gradients
    - Optionally clip gradients: nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    - Call optimiser.step() to update weights
    
    TENSOR OPERATIONS YOU'LL NEED:
    - torch.as_tensor(array, device=DEVICE, dtype=torch.float32)
    - tensor.unsqueeze(1) to add dimension
    - tensor.gather(1, indices) to select specific values
    - tensor.max(1)[0] to get maximum along dimension 1
    - tensor.squeeze() to remove extra dimensions
    
    RETURN TYPE: None (this function modifies policy network weights in-place)
    """
    
    # TODO: Implement DQN optimization step
    # Check if enough samples are available
    # Not enough samples to form a batch
    if len(memory) < batch_size:
        return

    # Sample batch of transitions: (state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    device = next(policy.parameters()).device

    # Convert to torch tensors on the correct device
    states = torch.as_tensor(states, device=device, dtype=torch.float32)
    actions = torch.as_tensor(actions, device=device, dtype=torch.int64).unsqueeze(1)
    rewards = torch.as_tensor(rewards, device=device, dtype=torch.float32).unsqueeze(1)
    next_states = torch.as_tensor(next_states, device=device, dtype=torch.float32)
    dones = torch.as_tensor(dones, device=device, dtype=torch.float32).unsqueeze(1)

    # Current Q-values for the actions actually taken: Q_policy(s, a)
    # policy(states): (B, n_actions) -> gather along action dimension
    q_values = policy(states).gather(1, actions)  # shape: (B, 1)

    # Double DQN target computation
    with torch.no_grad():
        # 1) Use policy network to select best action at next state: a* = argmax_a Q_policy(s', a)
        next_q_policy = policy(next_states)                          # (B, n_actions)
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)     # (B, 1)

        # 2) Use target network to evaluate that action: Q_target(s', a*)
        next_q_target = target(next_states)                          # (B, n_actions)
        next_q_values = next_q_target.gather(1, next_actions)        # (B, 1)

        # 3) Bellman target: r + (1 - done) * gamma * Q_target(s', a*)
        target_q_values = rewards + (1.0 - dones) * gamma * next_q_values

    # Mean Squared Error loss between current Q and target Q
    loss = nn.functional.mse_loss(q_values, target_q_values)

    # Optimize the policy network
    optimiser.zero_grad()
    loss.backward()

    # Optional: gradient clipping to improve stability
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

    optimiser.step()

    return loss.item()
