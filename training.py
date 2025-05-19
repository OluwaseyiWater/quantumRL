import gymnasium as gym
import numpy as np
import scipy.io as sio

from classicalRL import ClassicalPolicyGradient
from quantumRL   import QuantumPolicyGradient  # QPPG
from qnpg        import QNPG
from qdqn        import QuantumDQN

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
num_episodes   = 500
print_interval = 50
seeds          = [42, 99, 123, 256, 512]

# Prepare storage for raw episode‐reward traces
all_results = {
    'CPG':   [],  # Classical REINFORCE
    'QNPG':  [],  # Quantum Natural PG
    'QPPG':  [],  # Your proposed method
    'Q-DQN': []   # Quantum DQN baseline
}
# ──────────────────────────────────────────────────────────────────────────────

for seed in seeds:
    np.random.seed(seed)

    # 1) Classical REINFORCE baseline (CPG)
    env_c = gym.make('ClassicalState-v1', noise_level=0.03, reward_type='fidelity')
    cpg   = ClassicalPolicyGradient(env_c, hidden_dim=64, lr=0.002, gamma=0.99)
    ep_c, avg_c, _ = cpg.train(num_episodes=num_episodes,
                               print_interval=print_interval,
                               seed=seed)
    all_results['CPG'].append(ep_c)

    # Common quantum environment
    env_q = gym.make('QuantumSingleQubit-v1', noise_level=0.03, reward_type='fidelity')

    # 2) Quantum Natural Policy Gradient (QNPG)
    qnpg = QNPG(env_q, hidden_dim=64, lr=0.002, gamma=0.99)
    ep_qnpg = qnpg.train(num_episodes=num_episodes,
                         print_interval=print_interval,
                         seed=seed)
    all_results['QNPG'].append(ep_qnpg)

    # 3) Quantum-Preconditioned PG (QPPG)
    qppg = QuantumPolicyGradient(env_q, hidden_dim=64, lr=0.002, gamma=0.99)
    ep_qppg = qppg.train(num_episodes=num_episodes,
                         print_interval=print_interval,
                         seed=seed)
    all_results['QPPG'].append(ep_qppg)

    # 4) Quantum DQN (value-based baseline)
    qdqn = QuantumDQN(env_q, lr=0.002, gamma=0.99)
    ep_qdqn = qdqn.train(num_episodes=num_episodes)
    all_results['Q-DQN'].append(ep_qdqn)

# ──────────────────────────────────────────────────────────────────────────────
# Stack into arrays of shape (num_seeds, num_episodes)
for key in all_results:
    all_results[key] = np.vstack(all_results[key])

# Save everything to a .mat for MATLAB plotting
sio.savemat('comparison4.mat', all_results)

print("Saved comparison4.mat with keys:", list(all_results.keys()))
# ──────────────────────────────────────────────────────────────────────────────
