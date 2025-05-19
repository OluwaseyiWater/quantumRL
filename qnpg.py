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
class QuantumPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, num_gates, hidden_dim=64, num_qubits=1, num_layers=1):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(obs_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,hidden_dim), nn.ReLU())
        self.gate_head = nn.Sequential(nn.Linear(hidden_dim,num_gates), nn.Softmax(dim=-1))
        self.theta_mean = nn.Linear(hidden_dim,1)
        self.theta_std = nn.Parameter(torch.tensor([0.5]))
        self.qc, self.q_params = SimpleStatevectorEstimator.create_pqc(num_qubits, num_layers, input_dim=obs_dim)
        self.param_values = nn.Parameter(torch.randn(len(self.q_params)))
        self.estimator = SimpleStatevectorEstimator(self.qc, self.q_params)

    def forward(self, obs):
        feats = self.feature(obs)
        exps = self.estimator.run(param_bindings={p: v.item() for p,v in zip(self.q_params,self.param_values)}, observables=[np.eye(2)])
        q_feats = torch.tensor(exps, dtype=torch.float32)
        logits = self.gate_head(feats) * q_feats
        return logits

    def sample_action(self, obs):
        obs_t = torch.FloatTensor(obs)
        logits = self.forward(obs_t)
        gate_dist = Categorical(logits)
        gate = gate_dist.sample().item()
        gate_log_prob = gate_dist.log_prob(torch.tensor(gate))
        feats = self.feature(obs_t)
        theta_mean = self.theta_mean(feats)
        theta_std = torch.exp(self.theta_std).expand_as(theta_mean)
        theta_dist = torch.distributions.Normal(theta_mean, theta_std)
        raw_theta = theta_dist.sample().item()
        theta = raw_theta % (2*math.pi)
        theta_log_prob = theta_dist.log_prob(torch.tensor(theta)).squeeze()
        action = {'gate_type':gate,'theta':np.array([theta],dtype=np.float32)}
        return action, gate_log_prob + theta_log_prob

class QNPG:
    def __init__(self, env, hidden_dim, lr, gamma):
        self.env=env
        obs_dim=env.observation_space.shape[0]
        num_gates=env.action_space['gate_type'].n
        self.policy=QuantumPolicyNetwork(obs_dim,num_gates,hidden_dim)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=lr)
        self.gamma=gamma

    def compute_returns(self, rewards):
        R=0; returns=[]
        for r in reversed(rewards): R=r+self.gamma*R; returns.insert(0,R)
        returns=torch.tensor(returns)
        if len(returns)>1: returns=(returns-returns.mean())/(returns.std()+1e-8)
        return returns

    def update_policy(self, log_probs, rewards):
        returns = self.compute_returns(rewards)
        loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=self.policy.parameters(),
            retain_graph=False,
            create_graph=False,
            allow_unused=True 
        )

      
        flat_grads = []
        for g, p in zip(grads, self.policy.parameters()):
            if g is None:
                flat_grads.append(torch.zeros(p.numel(), device=p.device))
            else:
                flat_grads.append(g.reshape(-1))
        grad_vec = torch.cat(flat_grads) 
        qfi_diag = torch.ones_like(grad_vec)
        precond = qfi_diag * grad_vec
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.grad = precond[idx:idx+n].view_as(p).clone()
            idx += n
        self.optimizer.step()

    def train(self, num_episodes=500, print_interval=25, seed=None):
        start=time.perf_counter()
        if seed is not None: np.random.seed(seed); self.env.reset(seed=seed)
        self.episode_rewards=[];self.avg_rewards=[]
        for ep in range(num_episodes):
            obs,_=self.env.reset(); done=False; log_probs=[]; rews=[]; total=0.0
            while not done:
                action,lp=self.policy.sample_action(obs)
                obs,r,done,_,_=self.env.step(action)
                total+=r; log_probs.append(lp); rews.append(r)
            self.update_policy(log_probs, rews)
            self.episode_rewards.append(total)
            avg=np.mean(self.episode_rewards[-print_interval:]) if ep+1>=print_interval else np.mean(self.episode_rewards)
            self.avg_rewards.append(avg)
            if (ep+1)%print_interval==0: print(f"[Quantum] Ep {ep+1}/{num_episodes}, Avg R: {avg:.4f}")
        print(f"[Quantum] Trained in {time.perf_counter()-start:.2f}s")
        return np.array(self.episode_rewards)
        sio.savemat('quantum_data2.mat', {
        'episode_rewards': np.array(agent.episode_rewards),
        'avg_rewards':     np.array(agent.avg_rewards)
        })

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
    def main(num_episodes=500, seed=None):
        # 1) set up and train
        env   = gym.make('QuantumSingleQubit-v1', noise_level=0.03, reward_type='fidelity')
        agent = QNPG(env, hidden_dim=64, lr=0.002, gamma=0.99)
        print("Starting quantum training...")
        ep_r = agent.train(num_episodes=num_episodes, print_interval=25, seed=seed)
        avg_r = agent.avg_rewards      # stored by train()

        # 2) plot
        agent.plot_performance()

        # 3) return exactly what we need
        return ep_r, avg_r

    # <<< UNPACK two arrays, not one >>>
    ep_r, avg_r = main()

    # 4) save them for MATLAB
    import scipy.io as sio, numpy as np
    sio.savemat('quantum_data2.mat', {
        'episode_rewards': np.array(ep_r),
        'avg_rewards':     np.array(avg_r)
    })

