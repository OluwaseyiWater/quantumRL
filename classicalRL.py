# %load quantumRL/classicalRL.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time  # Import the time module directly

# --------------------------------
# Part 1: Classical Environment
# --------------------------------
class ClassicalStateEnv(gym.Env):
    """
    Classical Reinforcement Learning Environment with a binary state vector.
    The agent's goal is to maximize the probability of the target state.
    Includes realistic noise models and task-specific rewards.
    """
    def __init__(self, max_steps=10, noise_level=0.05, reward_type='measurement'):
        super().__init__()
        
        # Define action space: discrete operations on state vector
        # [operation_type, intensity]
        # operation_type: 0=Shift, 1=Scale, 2=Rotate, 3=Flip, 4=Identity
        self.action_space = spaces.Dict({
            'operation_type': spaces.Discrete(5),
            'intensity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        # Observation space: State vector and last measurement
        # [state_0, state_1, last_measurement]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # State initialization (probability distribution)
        self.state = np.array([1.0, 0.0], dtype=np.float32)  # Initial state
        self.target_state = np.array([0.0, 1.0], dtype=np.float32)  # Target state
        
        # Noise model parameters
        self.noise_level = noise_level
        self.reward_type = reward_type
        
        # Episode control
        self.max_steps = max_steps
        self.current_step = 0
        self.last_measurement = 0  # Store last measurement outcome
        
    def apply_operation(self, operation_type, intensity):
        """Apply parameterized operation to the state vector"""
        # Ensure state is normalized
        self.state = self.state / np.sum(self.state)
        
        if operation_type == 0:  # Shift operation
            # Shift probability mass
            delta = intensity * self.state[0]
            self.state[0] -= delta
            self.state[1] += delta
            
        elif operation_type == 1:  # Scale operation
            # Scale relative probabilities
            factor = 1.0 + intensity
            self.state[1] *= factor
            
        elif operation_type == 2:  # Rotate operation
            # Rotate probabilities (similar to quantum rotation)
            theta = intensity * np.pi
            new_state = np.zeros_like(self.state)
            new_state[0] = self.state[0] * np.cos(theta) - self.state[1] * np.sin(theta)
            new_state[1] = self.state[0] * np.sin(theta) + self.state[1] * np.cos(theta)
            # Ensure non-negative by taking absolute values
            new_state = np.abs(new_state)
            self.state = new_state
            
        elif operation_type == 3:  # Flip operation
            # Flip the state probabilities
            self.state = np.flip(self.state)
            
        else:  # Identity operation
            # Do nothing
            pass
            
        # Ensure state remains a valid probability distribution
        self.state = np.clip(self.state, 0.0, 1.0)
        sum_state = np.sum(self.state)
        if sum_state == 0.0:
            self.state = np.array([0.5, 0.5], dtype=np.float32)
        else:
            self.state = self.state / sum_state
    
    def apply_noise(self):
        """Apply noise to the state vector"""
        # Random fluctuation
        if np.random.random() < self.noise_level:
            # Apply random flip with some probability
            self.state = np.flip(self.state)
        
        # Random drift noise
        drift = self.noise_level * np.random.uniform(-0.1, 0.1)
        if self.state[0] + drift > 0 and self.state[1] - drift > 0:
            self.state[0] += drift
            self.state[1] -= drift
        
        # Ensure state remains valid
        self.state = np.clip(self.state, 0.0, 1.0)
        sum_state = np.sum(self.state)
        if sum_state == 0.0:
            self.state = np.array([0.5, 0.5], dtype=np.float32)
        else:
            self.state = self.state / sum_state
    
    def perform_measurement(self):
        """Perform a measurement and collapse the state"""
        # Sample from the probability distribution
        outcome = np.random.choice([0, 1], p=self.state)
        
        # Collapse state based on measurement
        if outcome == 0:
            self.state = np.array([1.0, 0.0], dtype=np.float32)
        else:
            self.state = np.array([0.0, 1.0], dtype=np.float32)
            
        self.last_measurement = outcome
        return outcome
    
    def calculate_reward(self):
        """Calculate reward based on the specified reward type"""
        if self.reward_type == 'measurement':
            # Reward based on measuring the target state
            return float(self.last_measurement)
        
        elif self.reward_type == 'fidelity':
            # Reward based on state preparation fidelity with target state
            # In classical version, this is dot product of probability distributions
            fidelity = np.sum(np.sqrt(self.state * self.target_state))**2
            return float(fidelity)
        
        elif self.reward_type == 'entropy':
            # Reward based on reducing entropy (maximizing purity)
            entropy = 0
            for p in self.state:
                if p > 0:
                    entropy -= p * np.log2(p)
            return 1.0 - entropy  # Max reward is 1 (pure state), min is 0 (mixed)
        
        else:
            return 0.0
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Reset to initial state
        self.state = np.array([1.0, 0.0], dtype=np.float32)
        self.current_step = 0
        self.last_measurement = 0
        
        # Set target state if provided in options
        if options and 'target_state' in options:
            self.target_state = options['target_state']
            
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Return the current observation"""
        # Return state vector and last measurement outcome
        return np.array([self.state[0], self.state[1], self.last_measurement], dtype=np.float32)
    
    def step(self, action):
        """Execute one step in the environment"""
        # Parse action
        operation_type = action['operation_type']
        intensity = action['intensity'].item()  # Use .item() instead of float()
        
        # Apply operation
        self.apply_operation(operation_type, intensity)
        
        # Apply noise model
        self.apply_noise()
        
        # Perform measurement with some probability
        measure_prob = 0.1  # Probability of measurement
        if np.random.random() < measure_prob:
            self.perform_measurement()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Update step count and check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Return observation, reward, termination flag
        return self._get_observation(), reward, terminated, False, {}
    
    def render(self):
        """Display state information"""
        print(f"Step: {self.current_step}")
        print(f"State: [{self.state[0]:.4f}, {self.state[1]:.4f}]")
        print(f"Last measurement: {self.last_measurement}")
        
    def close(self):
        pass

# Register environment
gym.register(id='ClassicalState-v1', entry_point=ClassicalStateEnv)

# --------------------------------
# Part 2: Policy Gradient Algorithm
# --------------------------------
class ClassicalPolicyNetwork(nn.Module):
    """
    Neural network for classical state manipulation policy.
    Input: state observation
    Output: probability distribution over operations and parameters
    """
    def __init__(self, obs_dim, num_operations, hidden_dim=64):
        super(ClassicalPolicyNetwork, self).__init__()
        
        # Common feature extractor
        self.feature_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Operation type head (discrete)
        self.operation_head = nn.Sequential(
            nn.Linear(hidden_dim, num_operations),
            nn.Softmax(dim=-1)
        )
        
        # Intensity parameter head (continuous)
        self.intensity_mean = nn.Linear(hidden_dim, 1)
        self.intensity_std = nn.Parameter(torch.tensor([0.1]))
        
    def forward(self, x):
        features = self.feature_network(x)
        
        # Operation probabilities
        operation_probs = self.operation_head(features)
        
        # Intensity parameter (mean and std for Gaussian policy)
        intensity_mean = torch.sigmoid(self.intensity_mean(features))  # Range [0,1]
        intensity_std = torch.exp(self.intensity_std).expand_as(intensity_mean)
        
        return operation_probs, intensity_mean, intensity_std
    
    def sample_action(self, obs):
        """Sample an action from the policy"""
        obs_tensor = torch.FloatTensor(obs)
        operation_probs, intensity_mean, intensity_std = self.forward(obs_tensor)
        
        # Sample operation type
        operation_distribution = Categorical(operation_probs)
        operation_type = operation_distribution.sample().item()
        
        # Sample intensity parameter (clipped to [0,1])
        intensity = torch.clamp(torch.normal(intensity_mean, intensity_std), 0.0, 1.0).item()
        
        # Calculate log probability for the action
        operation_log_prob = operation_distribution.log_prob(torch.tensor(operation_type))
        
        # Normal PDF for continuous intensity (scaled for clipping)
        intensity_log_prob = -0.5 * (((torch.tensor(intensity) - intensity_mean) / intensity_std) ** 2) - \
                             torch.log(intensity_std) - 0.5 * np.log(2 * np.pi)
        
        total_log_prob = operation_log_prob + intensity_log_prob.squeeze()
        
        return {
            'operation_type': operation_type,
            'intensity': np.array([intensity], dtype=np.float32)
        }, total_log_prob
    

class ClassicalPolicyGradient:
    """
    Policy Gradient algorithm for classical state manipulation tasks
    """
    def __init__(self, env, hidden_dim=64, lr=0.001, gamma=0.99):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.num_operations = env.action_space['operation_type'].n
        
        # Policy network
        self.policy = ClassicalPolicyNetwork(self.obs_dim, self.num_operations, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.episode_lengths = []
        self.actions_history = []
        self.intensity_history   = []
        
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # Normalize returns
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self, log_probs, rewards):
        """Update policy using policy gradient"""
        returns = self.compute_returns(rewards)
        
        # Calculate loss - fix tensor dimensions
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            # Ensure log_prob is a proper tensor
            if not isinstance(log_prob, torch.Tensor):
                log_prob = torch.tensor(log_prob)
            
            # Create proper loss tensor
            policy_loss.append(-log_prob * R)
        
        if policy_loss:
            # Stack scalars into 1D tensor
            policy_loss = torch.stack(policy_loss).sum()
            
            # Perform backpropagation
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
    def train(self, num_episodes=500, max_steps=50, print_interval=10):
        """Train the agent"""
        start_time = time.perf_counter()  # Fixed - use Python's time module
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            log_probs = []
            rewards = []
            episode_reward = 0
            length = 0
            first_step = True
            
            # Collect trajectory
            for step in range(max_steps):
                # Sample action from policy
                action, log_prob = self.policy.sample_action(obs)
                self.actions_history.append(action['operation_type'])
                length += 1
                if first_step:
                    # record the intensity value (scalar)
                    self.intensity_history.append(action['intensity'][0])
                    first_step = False
                
                # Execute action
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store log probability and reward
                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward
                
                # Move to next state
                obs = next_obs
                
                if done:
                    break
            
            # Update policy
            self.update_policy(log_probs, rewards)
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(length)
            
            # Calculate moving average
            if len(self.episode_rewards) >= 100:
                avg_reward = np.mean(self.episode_rewards[-100:])
                self.avg_rewards.append(avg_reward)
            else:
                self.avg_rewards.append(np.mean(self.episode_rewards))
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {self.avg_rewards[-1]:.4f}")
        
        end_time = time.perf_counter()  # Fixed - use Python's time module
        total_time = end_time - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        return total_time
    
    def plot_performance(self):
        """Plot training performance"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.3, label='Episode Rewards')
        plt.plot(self.avg_rewards, label='Avg Reward (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Classical Policy Gradient Training Progress')
        plt.legend()
        plt.grid()
        plt.savefig('classical_training_progress.png', dpi=600, bbox_inches='tight')
        plt.show()


            # ---- Episode Length Over Training ----
        plt.figure(figsize=(8,4))
        plt.plot(self.episode_lengths, alpha=0.3, label='Steps per Episode')
        plt.plot(
            np.convolve(self.episode_lengths, np.ones(50)/50, mode='valid'),
            label='Moving Avg (50 eps)',
        )
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Length Over Training')
        plt.legend()
        plt.grid()
        plt.savefig('fig_classical_episode_length.png', dpi=300, bbox_inches='tight')
        plt.show()

            # ---- Action Selection Frequency ----
        from collections import Counter
        ctr = Counter(self.actions_history)
        acts, cnts = zip(*sorted(ctr.items()))
        labels = ["Shift","Scale","Rotate","Flip","Identity"]
        plt.figure(figsize=(6,3))
        plt.bar(acts, cnts, tick_label=labels)
        plt.xlabel('Operation Type')
        plt.ylabel('Total Selections')
        plt.title('Action Selection Frequency')
        plt.grid(axis='y')
        plt.savefig('fig_classical_action_freq.png', dpi=300, bbox_inches='tight')
        plt.show()

            # ---- Parameter Convergence: First Intensity ----
        plt.figure(figsize=(8,4))
        plt.plot(self.intensity_history, alpha=0.5)
        plt.xlabel('Episode')
        plt.ylabel('First Intensity Value')
        plt.title('Convergence of First-Step Intensity Parameter')
        plt.grid()
        plt.savefig('fig_classical_intensity_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()



    
    def evaluate(self, num_episodes=20):
        """Evaluate the trained policy"""
        eval_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            print(f"\nEvaluating Episode {episode+1}")
            
            while not done and step < 50:
                # Sample action from policy (no gradients needed)
                with torch.no_grad():
                    action, _ = self.policy.sample_action(obs)
                
                # Execute action
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Print step information
                operation_names = ["Shift", "Scale", "Rotate", "Flip", "Identity"]
                operation_name = operation_names[action['operation_type']]
                print(f"  Step {step+1}: Operation={operation_name}({action['intensity'][0]:.2f}), Reward={reward:.4f}")
                
                # Update metrics
                episode_reward += reward
                obs = next_obs
                step += 1
                
                # Optional: render environment
                self.env.render()
                
            print(f"Episode {episode+1} Total Reward: {episode_reward:.4f}")
            eval_rewards.append(episode_reward)
            
        print(f"\nAverage Evaluation Reward: {np.mean(eval_rewards):.4f}")
        return np.mean(eval_rewards)
    
    def execute_sequence(self, render=True):
        """Execute a full sequence using the learned policy"""
        obs, _ = self.env.reset()
        done = False
        step = 0
        sequence = []
        
        print("\nExecuting Optimal Sequence with Learned Policy:")
        
        while not done and step < 10:  # Limit to 10 operations for readability
            # Sample action from policy
            with torch.no_grad():
                action, _ = self.policy.sample_action(obs)
            
            # Map operation type to name
            operation_names = ["Shift", "Scale", "Rotate", "Flip", "Identity"]
            operation_name = operation_names[action['operation_type']]
            intensity = action['intensity'][0]
            
            # Add to sequence
            sequence.append((operation_name, intensity))
            
            # Execute action
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Print step information
            if render:
                print(f"Operation {step+1}: {operation_name}({intensity:.2f})")
                self.env.render()
            
            # Move to next state
            obs = next_obs
            step += 1
        
        # Print final state and sequence
        print("\nFinal Sequence:")
        for i, (operation, param) in enumerate(sequence):
            print(f"{i+1}. {operation}({param:.2f})")
        
        return sequence

# --------------------------------
# Part 3: Main execution
# --------------------------------
def main():
    """Main function to run the classical RL experiment"""
    # Create environment
    env = gym.make('ClassicalState-v1', noise_level=0.03, reward_type='fidelity')
    
    # Create and train agent
    agent = ClassicalPolicyGradient(env, hidden_dim=64, lr=0.002, gamma=0.99)
    
    print("Starting training...")
    training_time = agent.train(num_episodes=500, print_interval=25)
    
    print(f"Training complete in {training_time:.2f} seconds. Plotting performance...")
    agent.plot_performance()
    
    print("Evaluating policy...")
    agent.evaluate(num_episodes=3)
    
    print("Executing optimal sequence...")
    agent.execute_sequence()
    return agent

if __name__ == "__main__":
    agent = main()

        # ---- Fidelity vs. Noise Level (Classical) ----
    import gymnasium as gym
    noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20]
    fidelities = []
    for eps in noise_levels:
        # swap in an env with higher noise
        env = gym.make('ClassicalState-v1', noise_level=eps, reward_type='fidelity')
        agent.env = env
        fidelities.append(agent.evaluate(num_episodes=50))

    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, fidelities, marker='o')
    plt.xlabel('Noise Level')
    plt.ylabel('Avg. Fidelity')
    plt.title('Final Fidelity vs. Environment Noise (Classical Env)')
    plt.grid()
    plt.savefig('fig_classical_fidelity_noise.png', dpi=300, bbox_inches='tight')
    plt.show()
