import numpy as np
import matplotlib.pyplot as plt

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
   
    agent_q = QuantumPolicyGradient(env_q, hidden_dim=64, lr=0.002, gamma=0.99)
    _ = agent_q.train(num_episodes=num_episodes, print_interval=print_interval)
    quantum_runs.append(np.array(agent_q.episode_rewards))

cl_data = np.vstack(classical_runs)
qt_data = np.vstack(quantum_runs)

eps    = np.arange(num_episodes)
mean_cl, std_cl = cl_data.mean(axis=0), cl_data.std(axis=0)
mean_qt, std_qt = qt_data.mean(axis=0), qt_data.std(axis=0)

plt.rcParams.update({
    'text.usetex':       False,      # disable external LaTeX
    'mathtext.fontset':  'stix',     # nicer math glyphs
    'font.family':       'serif',    # optional: more LaTeX-like look
    'font.size':         14,         # base font size for text
    'axes.titlesize':    18,         # title
    'axes.labelsize':    16,         # x/y labels
    'legend.fontsize':   14,         # legend entries
    'xtick.labelsize':   12,         # x-axis tick labels
    'ytick.labelsize':   12,         # y-axis tick labels
})

fig, ax = plt.subplots(figsize=(10,5))

# classical baseline
ax.plot(eps, mean_cl, label=r'Baseline (Classical REINFORCE)')
ax.fill_between(eps,
                mean_cl - std_cl,
                mean_cl + std_cl,
                alpha=0.3)

# proposed quantum method
ax.plot(eps, mean_qt, label=r'Proposed QPPG')
ax.fill_between(eps,
                mean_qt - std_qt,
                mean_qt + std_qt,
                alpha=0.3)

ax.set_title('Classical vs. Quantum Policy Gradient: Reward over Episodes')
ax.set_xlabel('Episode', fontsize=16)
ax.set_ylabel('Reward',  fontsize=16)
ax.legend(loc='best')
ax.grid(True)

fig.tight_layout()
fig.savefig('fig_comparison_avg.png', dpi=800)
plt.show()

print("\nSaved comparison plot as fig_comparison_avg.png")


