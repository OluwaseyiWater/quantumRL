
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import math
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector

# --------------------------------
# Part 1: Quantum Environment
# --------------------------------
class SimpleStatevectorEstimator:
    def __init__(self, qc, q_params):
        self.qc, self.q_params = qc, list(q_params)

    def run(self, circuit=None, param_bindings=None, observables=None):
        qc_bound = self.qc.assign_parameters(param_bindings)
        sv = Statevector.from_instruction(qc_bound)
        vec = sv.data
        return [np.real(vec.conj() @ (H @ vec)) for H in observables]

    @staticmethod
    def create_pqc(num_qubits, num_layers, input_dim):
        qc = QuantumCircuit(num_qubits)
        data_params = ParameterVector('x', length=input_dim * num_qubits)
        idx = 0
        for i in range(input_dim):
            for q in range(num_qubits):
                qc.ry(data_params[idx], q)
                idx += 1
        var_params = ParameterVector('θ', length=num_layers * num_qubits * 3)
        vp_idx = 0
        for _ in range(num_layers):
            for q in range(num_qubits):
                qc.rx(var_params[vp_idx], q); vp_idx += 1
                qc.ry(var_params[vp_idx], q); vp_idx += 1
                qc.rz(var_params[vp_idx], q); vp_idx += 1
            if num_qubits > 1:
                for q in range(num_qubits - 1):
                    qc.cx(q, q+1)
                qc.cx(num_qubits - 1, 0)
        return qc, list(var_params) + list(data_params)

class QuantumSingleQubitEnv(gym.Env):
    def __init__(self, max_steps=10, noise_level=0.05, reward_type='measurement'):
        super().__init__()
        self.action_space = spaces.Dict({
            'gate_type': spaces.Discrete(5),
            'theta': spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32)
        })
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.state = np.array([1.0, 0.0], dtype=np.complex64)
        self.target_state = np.array([0.0, 1.0], dtype=np.complex64)
        self.noise_level = noise_level
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.current_step = 0
        self.last_measurement = 0

    def apply_gate(self, gate_type, theta=0):
        if gate_type == 0:
            return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex64)
        elif gate_type == 1:
            return np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex64)
        elif gate_type == 2:
            return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype=np.complex64)
        elif gate_type == 3:
            return np.array([[1/np.sqrt(2), 1/np.sqrt(2)],[1/np.sqrt(2), -1/np.sqrt(2)]], dtype=np.complex64)
        else:
            return np.eye(2, dtype=np.complex64)

    def apply_noise(self):
        if np.random.random() < self.noise_level:
            self.state = np.dot(np.array([[0,1],[1,0]], dtype=np.complex64), self.state)
        gamma = self.noise_level
        if np.random.random() < gamma * np.abs(self.state[1])**2:
            self.state = np.array([1.0, 0.0], dtype=np.complex64)
        self.state /= np.linalg.norm(self.state)
        if np.random.random() < self.noise_level:
            phase = np.random.uniform(0, 2*np.pi)
            self.state = np.dot(np.array([[1,0],[0,np.exp(1j*phase)]], dtype=np.complex64), self.state)
        self.state /= np.linalg.norm(self.state)

    def perform_measurement(self):
        prob_1 = np.clip(np.abs(self.state[1])**2, 0.0, 1.0)
        prob_0 = 1.0 - prob_1
        outcome = np.random.choice([0,1], p=[prob_0, prob_1])
        self.state = np.array([1.0,0.0], dtype=np.complex64) if outcome==0 else np.array([0.0,1.0], dtype=np.complex64)
        self.last_measurement = outcome
        return outcome

    def calculate_reward(self):
        if self.reward_type=='measurement': return float(self.last_measurement)
        if self.reward_type=='fidelity': return float(np.abs(np.vdot(self.state,self.target_state))**2)
        prob0, prob1 = np.abs(self.state)**2
        entropy=0
        for p in [prob0,prob1]:
            if p>0: entropy -= p*np.log2(p)
        return 1.0-entropy

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.state = np.array([1.0,0.0], dtype=np.complex64)
        self.current_step=0; self.last_measurement=0
        return np.array([1.0,0.0,0],dtype=np.float32),{}

    def step(self, action):
        gate = self.apply_gate(action['gate_type'], float(action['theta']))
        self.state = gate @ self.state
        self.apply_noise()
        if np.random.random()<0.1: self.perform_measurement()
        reward = self.calculate_reward()
        self.current_step +=1
        done = self.current_step>=self.max_steps
        return np.array([np.abs(self.state[0])**2, np.abs(self.state[1])**2, self.last_measurement],dtype=np.float32), reward, done, False, {}

gym.register(id='QuantumSingleQubit-v1', entry_point=QuantumSingleQubitEnv)

# --------------------------------
# Part 2: Policy Gradient Algorithm
# --------------------------------
class QuantumQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, num_qubits=1, num_layers=2):
        super().__init__()
        self.qc, self.q_params = SimpleStatevectorEstimator.create_pqc(
            num_qubits, num_layers, input_dim=obs_dim)
        self.param_values = nn.Parameter(torch.randn(len(self.q_params)))
        self.estimator = SimpleStatevectorEstimator(self.qc, self.q_params)
        self.fc_out = nn.Linear(num_qubits, action_dim)

    def forward(self, state):
        batch_size = state.shape[0]
        q_out = []
        for obs in state:
            q_feat = self.estimator.run(param_bindings={p: v.item() for p, v in zip(self.q_params, self.param_values)},
                                        observables=[np.eye(2)])
            q_out.append(q_feat)
        q_out = torch.tensor(q_out, dtype=torch.float32)
        return self.fc_out(q_out)

class QuantumDQN:
    def __init__(self, env, lr=1e-3, gamma=0.99, replay_capacity=10000):
        self.env = env
        obs_dim   = env.observation_space.shape[0]
        act_dim   = env.action_space['gate_type'].n
        self.online_net = QuantumQNetwork(obs_dim, act_dim)
        self.target_net = QuantumQNetwork(obs_dim, act_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = []  # simple list; replace with deque for efficiency
        self.capacity = replay_capacity

    def select_action(self, state, eps):
        if np.random.rand() < eps:
            return self.env.action_space.sample()
        qs = self.online_net(torch.FloatTensor(state).unsqueeze(0))
        a  = qs.argmax(dim=1).item()
        return {'gate_type': a, 'theta': np.array([0.0],dtype=np.float32)}

    def store(self, transition):
        self.replay.append(transition)
        if len(self.replay) > self.capacity:
            self.replay.pop(0)

    def sample_batch(self, batch_size):
        idx = np.random.choice(len(self.replay), batch_size, replace=False)
        batch = [self.replay[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def update(self, batch_size=32):
        if len(self.replay) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        # Q(s,a)
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        # max_a' Q_target(s',a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes=500, batch_size=32, eps_start=1.0, eps_end=0.01, eps_decay=0.995, target_update=10):
        eps = eps_start
        ep_rewards = []
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_r = 0
            while not done:
                action = self.select_action(state, eps)
                next_s, r, done, _, _ = self.env.step(action)
                total_r += r
                a_idx = action['gate_type']
                self.store((state, a_idx, r, next_s, float(done)))
                state = next_s
                self.update(batch_size)
            ep_rewards.append(total_r)
            eps = max(eps*eps_decay, eps_end)
            if (ep+1) % target_update == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            if (ep+1) % 50 == 0:
                print(f"Q-DQN Ep {ep+1}, R={total_r:.3f}, eps={eps:.2f}")
        return np.array(ep_rewards)
        
    def plot_performance(self):
        plt.rcParams['text.usetex']    = False
        plt.rcParams['mathtext.fontset'] = 'stix'  
        plt.rcParams['font.size']      = 14         # base font size
        plt.rcParams['axes.labelsize'] = 16         # axes labels
        plt.rcParams['axes.titlesize'] = 18         # title
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        plt.figure(figsize=(10,5))
        plt.plot(self.episode_rewards,alpha=0.3,label=r'Episode Reward')
        plt.plot(self.avg_rewards,label=r'Avg Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Quantum Training Progress')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('quantum_training_progress.png',dpi=600,bbox_inches='tight');
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

if __name__=='__main__':
    def main(num_episodes=500):
        env   = gym.make('QuantumSingleQubit-v1',
                         noise_level=0.03,
                         reward_type='fidelity')
        agent = QuantumDQN(env,
                           lr=0.002,
                           gamma=0.99)   

        print("Starting Q-DQN training…")
        ep_r = agent.train(num_episodes=num_episodes)

        window = 25
        avg_r = np.array([ ep_r[max(0,i-window+1):i+1].mean()
                           for i in range(len(ep_r)) ])

        return ep_r, avg_r

    ep_r, avg_r = main()

    # 5) save to .mat for MATLAB
    sio.savemat('quantum_data3.mat', {
        'episode_rewards': ep_r,
        'avg_rewards':     avg_r
    })
