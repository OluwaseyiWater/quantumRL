import numpy as np
import matplotlib.pyplot as plt

# static imports — no more FileNotoundError
from classicalRL import ClassicalStateEnv, ClassicalPolicyGradient
from quantumRL   import QuantumSingleQubitEnv, QuantumPolicyGradient

# reproducible seeds
seeds = [42, 99, 123, 256, 512]

num_episodes   = 500
print_interval = 50

classical_runs = []
quantum_runs   = []

for seed in seeds:
    print(f"\n=== Seed {seed}: Classical ===")
    np.random.seed(seed)
    env_c = ClassicalStateEnv(noise_level=0.03, reward_type='fidelity')
    # classical.train returns (ep_rewards, avg_rewards, duration)
    ep_r_c, avg_r_c, _ = ClassicalPolicyGradient(
        env_c, hidden_dim=64, lr=0.002, gamma=0.99
    ).train(
        num_episodes=num_episodes,
        print_interval=print_interval,
        seed=seed
    )
    classical_runs.append(np.array(ep_r_c))

    print(f"\n=== Seed {seed}: Quantum ===")
    np.random.seed(seed)
    env_q = QuantumSingleQubitEnv(noise_level=0.03, reward_type='fidelity')
    # quantum.train only returns duration, but stores episode_rewards
    agent_q = QuantumPolicyGradient(env_q, hidden_dim=64, lr=0.002, gamma=0.99)
    _ = agent_q.train(num_episodes=num_episodes, print_interval=print_interval)
    quantum_runs.append(np.array(agent_q.episode_rewards))

# turn to arrays: shape (5, 500)
cl_data = np.vstack(classical_runs)
qt_data = np.vstack(quantum_runs)

eps    = np.arange(num_episodes)
mean_cl, std_cl = cl_data.mean(axis=0), cl_data.std(axis=0)
mean_qt, std_qt = qt_data.mean(axis=0), qt_data.std(axis=0)

plt.figure(figsize=(10,5))
# classical
plt.plot(eps, mean_cl, label='Classical PG')
plt.fill_between(eps, mean_cl - std_cl, mean_cl + std_cl, alpha=0.3)
# quantum
plt.plot(eps, mean_qt, label='Quantum PG')
plt.fill_between(eps, mean_qt - std_qt, mean_qt + std_qt, alpha=0.3)

plt.title('Classical vs. Quantum Policy Gradient: Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('fig_comparison_avg.png', dpi=300)
plt.show()

print("\nSaved comparison plot as fig_comparison_avg.png")

