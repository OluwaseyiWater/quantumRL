import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time  # Import the time module directly


class ClassicalStateEnv(gym.Env):
    """
    Classical Reinforcement Learning Environment with a binary state vector.
    """
    def __init__(self, max_steps=10, noise_level=0.05, reward_type='measurement'):
        super().__init__()
        self.action_space = gym.spaces.Dict({
            'operation_type': gym.spaces.Discrete(5),
            'intensity': gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        })
        self.observation_space = gym.spaces.Box(0.0, 1.0, (3,), dtype=np.float32)
        self.state = np.array([1.0, 0.0], dtype=np.float32)
        self.target_state = np.array([0.0, 1.0], dtype=np.float32)
        self.noise_level = noise_level
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.current_step = 0
        self.last_measurement = 0

    def apply_operation(self, operation_type, intensity):
        self.state = self.state / np.sum(self.state)
        if operation_type == 0:
            delta = intensity * self.state[0]
            self.state[0] -= delta; self.state[1] += delta
        elif operation_type == 1:
            factor = 1.0 + intensity; self.state[1] *= factor
        elif operation_type == 2:
            theta = intensity * np.pi
            new = np.array([
                self.state[0]*np.cos(theta) - self.state[1]*np.sin(theta),
                self.state[0]*np.sin(theta) + self.state[1]*np.cos(theta)
            ], dtype=np.float32)
            self.state = np.abs(new)
        elif operation_type == 3:
            self.state = np.flip(self.state)
        # identity does nothing
        self.state = np.clip(self.state, 0, 1)
        s = np.sum(self.state)
        self.state = self.state / s if s > 0 else np.array([0.5, 0.5], dtype=np.float32)

    def apply_noise(self):
        if np.random.random() < self.noise_level:
            self.state = np.flip(self.state)
        drift = self.noise_level * np.random.uniform(-0.1, 0.1)
        if self.state[0] + drift > 0 and self.state[1] - drift > 0:
            self.state[0] += drift; self.state[1] -= drift
        self.state = np.clip(self.state, 0, 1)
        s = np.sum(self.state)
        self.state = self.state / s if s > 0 else np.array([0.5, 0.5], dtype=np.float32)

    def perform_measurement(self):
        outcome = np.random.choice([0, 1], p=self.state)
        self.state = np.array([1.0, 0.0], dtype=np.float32) if outcome == 0 else np.array([0.0, 1.0], dtype=np.float32)
        self.last_measurement = outcome
        return outcome

    def calculate_reward(self):
        if self.reward_type == 'measurement':
            return float(self.last_measurement)
        if self.reward_type == 'fidelity':
            fid = np.sum(np.sqrt(self.state * self.target_state))**2
            return float(fid)
        if self.reward_type == 'entropy':
            ent = -np.sum([p * np.log2(p) for p in self.state if p > 0])
            return 1.0 - ent
        return 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.array([1.0, 0.0], dtype=np.float32)
        self.current_step = 0
        self.last_measurement = 0
        if options and 'target_state' in options:
            self.target_state = options['target_state']
        return np.array([self.state[0], self.state[1], self.last_measurement], dtype=np.float32), {}

    def step(self, action):
        op = action['operation_type']
        intensity = action['intensity'].item()
        self.apply_operation(op, intensity)
        self.apply_noise()
        if np.random.random() < 0.1:
            self.perform_measurement()
        r = self.calculate_reward()
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return np.array([self.state[0], self.state[1], self.last_measurement], dtype=np.float32), r, done, False, {}

    def render(self, mode='human'):
        print(f"Step {self.current_step}: state={self.state}, last_measurement={self.last_measurement}")
    def close(self): pass

gym.register('ClassicalState-v1', entry_point=ClassicalStateEnv)


class ClassicalPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, num_ops, hidden_dim=64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim, num_ops), nn.Softmax(dim=-1)
        )
        self.int_mean = nn.Linear(hidden_dim, 1)
        self.int_std = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        f = self.feature(x)
        probs = self.op_head(f)
        mean = torch.sigmoid(self.int_mean(f))
        std = torch.exp(self.int_std).expand_as(mean)
        return probs, mean, std

    def sample_action(self, obs):
        obs_t = torch.FloatTensor(obs)
        probs, mean, std = self.forward(obs_t)
        dist = Categorical(probs)
        op = dist.sample().item()
        intensity = torch.clamp(torch.normal(mean, std), 0, 1).item()
        logp = dist.log_prob(torch.tensor(op)) + (
            -0.5 * ((intensity - mean) / std)**2
            - torch.log(std)
            - 0.5 * np.log(2 * np.pi)
        ).squeeze()
        return {'operation_type': op, 'intensity': np.array([intensity], dtype=np.float32)}, logp


class ClassicalPolicyGradient:
    def __init__(self, env, hidden_dim, lr, gamma):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        num_ops = env.action_space['operation_type'].n
        self.policy = ClassicalPolicyNetwork(obs_dim, num_ops, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update_policy(self, log_probs, rewards):
        returns = self.compute_returns(rewards)
        loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes=500, print_interval=25, seed=None):
        """
        Train the agent, measure elapsed time, and return
        (episode_rewards, avg_rewards, duration).
        """
        start = time.perf_counter()
        if seed is not None:
            np.random.seed(seed)
            self.env.reset(seed=seed)

        ep_rewards = []
        avg_rewards = []
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total = 0.0
            logp = []
            rews = []
            while not done:
                action, lp = self.policy.sample_action(obs)
                obs, r, done, _, _ = self.env.step(action)
                total += r
                logp.append(lp)
                rews.append(r)

            self.update_policy(logp, rews)
            ep_rewards.append(total)

            if ep + 1 >= print_interval:
                avg = np.mean(ep_rewards[-print_interval:])
            else:
                avg = np.mean(ep_rewards)
            avg_rewards.append(avg)

            if (ep + 1) % print_interval == 0:
                print(f"Episode {ep+1}/{num_episodes}, Avg Reward: {avg:.4f}")

        duration = time.perf_counter() - start
        print(f"Training completed in {duration:.2f} seconds")

        # Store for later use
        self.episode_rewards = ep_rewards
        self.avg_rewards = avg_rewards

        # Return full training history plus duration
        return ep_rewards, avg_rewards, duration


    def plot_performance(self):
        plt.rcParams['text.usetex']    = False
        plt.rcParams['mathtext.fontset'] = 'stix'  
        plt.rcParams['font.size']      = 14         
        plt.rcParams['axes.labelsize'] = 16         
        plt.rcParams['axes.titlesize'] = 18        
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.3, label=r'Episode Reward')
        plt.plot(self.avg_rewards, label=r'Avg Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('classical_training_progress.png', dpi=600, bbox_inches='tight')
        plt.show()


    def evaluate(self, num_episodes=20):
        eval_rewards = []
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            print(f"\nEvaluating Episode {episode+1}")
            while not done and step < 50:
                with torch.no_grad():
                    action, _ = self.policy.sample_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                operation_names = ["Shift","Scale","Rotate","Flip","Identity"]
                op_name = operation_names[action['operation_type']]
                print(f"  Step {step+1}: Operation={op_name}({action['intensity'][0]:.2f}), Reward={reward:.4f}")
                episode_reward += reward
                obs = next_obs
                step += 1
            print(f"Episode {episode+1} Total Reward: {episode_reward:.4f}")
            eval_rewards.append(episode_reward)
        print(f"\nAverage Evaluation Reward: {np.mean(eval_rewards):.4f}")
        return np.mean(eval_rewards)

    def execute_sequence(self, render=True):
        obs, _ = self.env.reset()
        done = False
        step = 0
        sequence = []
        print("\nExecuting Optimal Sequence with Learned Policy:")
        while not done and step < 10:
            with torch.no_grad():
                action, _ = self.policy.sample_action(obs)
            operation_names = ["Shift","Scale","Rotate","Flip","Identity"]
            op_name = operation_names[action['operation_type']]
            intensity = action['intensity'][0]
            sequence.append((op_name, intensity))
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if render:
                print(f"Operation {step+1}: {op_name}({intensity:.2f})")
                self.env.render()
            obs = next_obs
            step += 1
        print("\nFinal Sequence:")
        for i, (op, param) in enumerate(sequence):
            print(f"{i+1}. {op}({param:.2f})")
        return sequence


def main():
    env = gym.make('ClassicalState-v1', noise_level=0.03, reward_type='fidelity')
    agent = ClassicalPolicyGradient(env, hidden_dim=64, lr=0.002, gamma=0.99)
    print("Starting training...")
    rewards, avg_rewards, duration = agent.train(num_episodes=500, print_interval=25)
    print(f"[Classical] Total training time: {duration:.2f}s")
    agent.plot_performance()
    agent.evaluate(num_episodes=3)
    agent.execute_sequence()
    return agent, rewards, avg_rewards, duration

if __name__ == "__main__":
    main()
