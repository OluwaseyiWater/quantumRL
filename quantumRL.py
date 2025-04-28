# %load quantumRL/quantumRL.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time

# --------------------------------
# Part 1: Quantum Environment
# --------------------------------
class QuantumSingleQubitEnv(gym.Env):
    """
    Quantum Reinforcement Learning Environment with a single qubit.
    The agent's goal is to maximize the probability of measuring |1⟩.
    Includes realistic NISQ device simulation with noise models, measurement collapse,
    task-specific rewards, and parametric gates.
    """
    def __init__(self, max_steps=10, noise_level=0.05, reward_type='measurement'):
        super().__init__()
        
        # Define action space: continuous parameters for rotation angles
        # [gate_type, theta, phi]
        # gate_type: 0=RX, 1=RY, 2=RZ, 3=H, 4=I
        self.action_space = spaces.Dict({
            'gate_type': spaces.Discrete(5),
            'theta': spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32)
        })
        
        # Observation space: Measurement outcomes (after collapse)
        # [prob_0, prob_1, last_measurement]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Quantum state initialization
        self.state = np.array([1.0, 0.0], dtype=np.complex64)  # |0⟩ state
        self.target_state = np.array([0.0, 1.0], dtype=np.complex64)  # |1⟩ state (default target)
        
        # Noise model parameters
        self.noise_level = noise_level
        self.reward_type = reward_type
        
        # Episode control
        self.max_steps = max_steps
        self.current_step = 0
        self.last_measurement = 0  # Store last measurement outcome
        
    def apply_gate(self, gate_type, theta=0):
        """Apply parameterized quantum gate"""
        if gate_type == 0:  # RX gate
            gate = np.array([
                [np.cos(theta/2), -1j*np.sin(theta/2)],
                [-1j*np.sin(theta/2), np.cos(theta/2)]
            ], dtype=np.complex64)
        elif gate_type == 1:  # RY gate
            gate = np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=np.complex64)
        elif gate_type == 2:  # RZ gate
            gate = np.array([
                [np.exp(-1j*theta/2), 0],
                [0, np.exp(1j*theta/2)]
            ], dtype=np.complex64)
        elif gate_type == 3:  # Hadamard
            gate = np.array([
                [1/np.sqrt(2), 1/np.sqrt(2)],
                [1/np.sqrt(2), -1/np.sqrt(2)]
            ], dtype=np.complex64)
        else:  # Identity
            gate = np.eye(2, dtype=np.complex64)
            
        return gate
    
    def apply_noise(self):
        """Apply depolarizing and amplitude damping noise"""
        # Depolarizing noise: mix with maximally mixed state
        if np.random.random() < self.noise_level:
            # Apply bit flip (X) with some probability
            self.state = np.dot(np.array([[0, 1], [1, 0]], dtype=np.complex64), self.state)
        
        # Amplitude damping (T1 decay)
        gamma = self.noise_level  # Damping parameter
        if np.random.random() < gamma * np.abs(self.state[1])**2:
            # Collapse to |0⟩ with probability proportional to |1⟩ population
            self.state = np.array([1.0, 0.0], dtype=np.complex64)

        """Apply depolarizing and amplitude damping noise"""
        # Ensure state remains valid
        self.state = self.state / np.linalg.norm(self.state)
        
        # Phase damping (T2 dephasing)
        if np.random.random() < self.noise_level:
            # Random phase rotation
            phase = np.random.uniform(0, 2*np.pi)
            phase_matrix = np.array([[1, 0], [0, np.exp(1j*phase)]], dtype=np.complex64)
            self.state = np.dot(phase_matrix, self.state)
        
        # Normalize state
        self.state = self.state / np.linalg.norm(self.state)
    
    def perform_measurement(self):
        """Perform a measurement and collapse the state"""
        prob_1 = np.clip(np.abs(self.state[1])**2, 0.0, 1.0)  # Ensure valid probability
        prob_0 = 1.0 - prob_1
        
        # Normalize probabilities to sum exactly to 1
        probabilities = np.array([prob_0, prob_1])
        probabilities /= probabilities.sum()
        
        outcome = np.random.choice([0, 1], p=probabilities)
        
        # Collapse state based on measurement
        if outcome == 0:
            self.state = np.array([1.0, 0.0], dtype=np.complex64)
        else:
            self.state = np.array([0.0, 1.0], dtype=np.complex64)
            
        self.last_measurement = outcome
        return outcome
    
    def calculate_reward(self):
        """Calculate reward based on the specified reward type"""
        if self.reward_type == 'measurement':
            # Reward based on measuring |1⟩
            return float(self.last_measurement)
        
        elif self.reward_type == 'fidelity':
            # Reward based on state preparation fidelity with target state
            fidelity = np.abs(np.vdot(self.state, self.target_state))**2
            return float(fidelity)
        
        elif self.reward_type == 'entropy':
            # Reward based on reducing entropy (maximizing purity)
            prob_0 = np.abs(self.state[0])**2
            prob_1 = np.abs(self.state[1])**2
            entropy = 0
            if prob_0 > 0:
                entropy -= prob_0 * np.log2(prob_0)
            if prob_1 > 0:
                entropy -= prob_1 * np.log2(prob_1)
            return 1.0 - entropy  # Max reward is 1 (pure state), min is 0 (mixed)
        
        else:
            return 0.0
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Reset to |0⟩ state
        self.state = np.array([1.0, 0.0], dtype=np.complex64)
        self.current_step = 0
        self.last_measurement = 0
        
        # Set target state if provided in options
        if options and 'target_state' in options:
            self.target_state = options['target_state']
            
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Return the current observation"""
        prob_0 = np.abs(self.state[0])**2
        prob_1 = np.abs(self.state[1])**2
        
        # Return probability distribution and last measurement outcome
        return np.array([prob_0, prob_1, self.last_measurement], dtype=np.float32)
    
    def step(self, action):
        """Execute one step in the environment"""
        # Parse action
        gate_type = action['gate_type']
        theta = float(action['theta'])
        
        # Apply quantum gate
        gate = self.apply_gate(gate_type, theta)
        self.state = np.dot(gate, self.state)
        
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
        """Display quantum state information"""
        prob_0 = np.abs(self.state[0])**2
        prob_1 = np.abs(self.state[1])**2
        
        print(f"Step: {self.current_step}")
        print(f"State: [{self.state[0]:.4f}, {self.state[1]:.4f}]")
        print(f"Probabilities: |0⟩: {prob_0:.4f}, |1⟩: {prob_1:.4f}")
        print(f"Last measurement: {self.last_measurement}")
        
    def close(self):
        pass

# Register environment
gym.register(id='QuantumSingleQubit-v1', entry_point=QuantumSingleQubitEnv)

# --------------------------------
# Part 2: Policy Gradient Algorithm
# --------------------------------
class QuantumPolicyNetwork(nn.Module):
    """
    Neural network for quantum gate policy.
    Input: quantum state observation
    Output: probability distribution over gates and parameters
    """
    def __init__(self, obs_dim, num_gates, hidden_dim=64):
        super(QuantumPolicyNetwork, self).__init__()
        
        # Common feature extractor
        self.feature_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Gate type head (discrete)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, num_gates),
            nn.Softmax(dim=-1)
        )
        
        # Theta parameter head (continuous)
        self.theta_mean = nn.Linear(hidden_dim, 1)
        self.theta_std = nn.Parameter(torch.tensor([0.5]))
        
    def forward(self, x):
        features = self.feature_network(x)
        
        # Gate probabilities
        gate_probs = self.gate_head(features)
        
        # Theta parameter (mean and std for Gaussian policy)
        theta_mean = self.theta_mean(features)
        theta_std = torch.exp(self.theta_std).expand_as(theta_mean)
        
        return gate_probs, theta_mean, theta_std
    
    def sample_action(self, obs):
        """Sample an action from the policy"""
        obs_tensor = torch.FloatTensor(obs)
        gate_probs, theta_mean, theta_std = self.forward(obs_tensor)
        
        # Sample gate type
        gate_distribution = Categorical(gate_probs)
        gate_type = gate_distribution.sample().item()
        
        # Sample theta parameter
        theta = torch.normal(theta_mean, theta_std).item()
        # Ensure theta is within [0, 2π]
        theta = theta % (2 * np.pi)
        
        # Calculate log probability for the action
        gate_log_prob = gate_distribution.log_prob(torch.tensor(gate_type))
        theta_log_prob = -0.5 * (((torch.tensor(theta) - theta_mean) / theta_std) ** 2) - \
                         torch.log(theta_std) - 0.5 * np.log(2 * np.pi)
        
        total_log_prob = gate_log_prob + theta_log_prob.squeeze()
        
        return {
            'gate_type': gate_type,
            'theta': np.array([theta], dtype=np.float32)
        }, total_log_prob
    

class QuantumPolicyGradient:
    """
    Policy Gradient algorithm for quantum control tasks
    """
    def __init__(self, env, hidden_dim=64, lr=0.001, gamma=0.99):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.num_gates = env.action_space['gate_type'].n
        
        # Policy network
        self.policy = QuantumPolicyNetwork(self.obs_dim, self.num_gates, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.episode_lengths = []
        self.actions_history = []
        self.theta_history   = []

    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # Normalize returns
        returns = torch.tensor(returns)
        if len(returns) > 1:  # Only normalize if we have more than one return
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
        

    def train(self, num_episodes=500, max_steps=50, print_interval=10):
        """Train the agent with time measurement"""
        start_time = time.perf_counter()
        
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
                self.actions_history.append(action['gate_type'])
                length += 1
                if first_step:
                    self.theta_history.append(action['theta'][0])
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
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        return total_time


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
    
    def plot_performance(self):
        """Plot training performance"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.3, label='Episode Rewards')
        plt.plot(self.avg_rewards, label='Avg Reward (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Quantum Policy Gradient Training Progress')
        plt.legend()
        plt.grid()
        plt.savefig('training_progress.png', dpi=600, bbox_inches='tight')
        plt.show()

    # --- after training and collecting agent.episode_lengths ---
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
        plt.savefig('fig_episode_length.png', dpi=300, bbox_inches='tight')
        plt.show()

        # ---- NEW: Action Selection Frequency ----
        from collections import Counter
        ctr = Counter(self.actions_history)
        acts, cnts = zip(*sorted(ctr.items()))
        labels = ["Rx","Ry","Rz","H","I"]
        plt.figure(figsize=(6,3))
        plt.bar(acts, cnts, tick_label=labels)
        plt.xlabel('Gate Type'); plt.ylabel('Selections')
        plt.title('Action Selection Frequency')
        plt.grid(axis='y')
        plt.savefig('fig_action_freq.png', dpi=300, bbox_inches='tight')
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
                gate_names = ["RX", "RY", "RZ", "H", "I"]
                gate_name = gate_names[action['gate_type']]
                print(f"  Step {step+1}: Gate={gate_name}({action['theta'][0]:.2f}), Reward={reward:.4f}")
                
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
    
    def execute_quantum_circuit(self, render=True):
        """Execute a full quantum circuit using the learned policy"""
        obs, _ = self.env.reset()
        done = False
        step = 0
        circuit = []
        
        print("\nExecuting Quantum Circuit with Learned Policy:")
        
        while not done and step < 10:  # Limit to 10 gates for readability
            # Sample action from policy
            with torch.no_grad():
                action, _ = self.policy.sample_action(obs)
            
            # Map gate type to name
            gate_names = ["RX", "RY", "RZ", "H", "I"]
            gate_name = gate_names[action['gate_type']]
            theta = action['theta'][0]
            
            # Add to circuit
            circuit.append((gate_name, theta))
            
            # Execute action
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Print step information
            if render:
                print(f"Gate {step+1}: {gate_name}({theta:.2f})")
                self.env.render()
            
            # Move to next state
            obs = next_obs
            step += 1
        
        # Print final state and circuit
        print("\nFinal Circuit:")
        for i, (gate, param) in enumerate(circuit):
            if gate in ["RX", "RY", "RZ"]:
                print(f"{i+1}. {gate}({param:.2f})")
            else:
                print(f"{i+1}. {gate}")
        
        return circuit

# --------------------------------
# Part 3: Main execution
# --------------------------------
def main():
    """Main function to run the quantum RL experiment"""
    # Create environment
    env = gym.make('QuantumSingleQubit-v1', noise_level=0.03, reward_type='fidelity')
    
    # Create and train agent
    agent = QuantumPolicyGradient(env, hidden_dim=64, lr=0.002, gamma=0.99)
    
    print("Starting training...")
    training_time = agent.train(num_episodes=500, print_interval=25)
    
    print(f"Training complete in {training_time:.2f} seconds. Plotting performance...")
    agent.plot_performance()
    
    print("Evaluating policy...")
    agent.evaluate(num_episodes=3)
    
    print("Executing optimal quantum circuit...")
    agent.execute_quantum_circuit()
    return agent

if __name__ == "__main__":
    agent = main()

    import gymnasium as gym
    noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20]
    fidelities = []
    for eps in noise_levels:
        # swap in a fresh env with higher noise
        env = gym.make('QuantumSingleQubit-v1', noise_level=eps, reward_type='fidelity')
        agent.env = env
        fidelities.append(agent.evaluate(num_episodes=50))

    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, fidelities, marker='o')
    plt.xlabel('Noise Level')
    plt.ylabel('Average Fidelity')
    plt.title('Final Fidelity vs. Environment Noise')
    plt.grid()
    plt.savefig('fig_fidelity_noise.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---- NEW: Parameter Convergence ----
    plt.figure(figsize=(8,4))
    plt.plot(agent.theta_history, alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('First Gate Angle (rad)')
    plt.title('Convergence of First Rotation Parameter')
    plt.grid()
    plt.savefig('fig_theta_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
