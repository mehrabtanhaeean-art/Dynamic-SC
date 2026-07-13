import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path
import time

random.seed(42)
np.random.seed(42)

sns.set_theme(style="white") 

try:
    excel_file = Path(__file__).with_name("P1.xlsx")
except NameError:
    excel_file = Path("P1.xlsx")

transition_mats = []
for a in range(1, 17):
    df = pd.read_excel(excel_file, sheet_name=f"A{a}", header=None)
    transition_mats.append(df.to_numpy())

mat_reward = pd.read_excel(excel_file, sheet_name="Reward", header=None).to_numpy()

num_states = 6
num_actions = 16

states_labels = ["S1", "S2", "S3", "S4", "S5", "S6"]
actions_labels = [f"A{k}" for k in range(1, 17)]

T = np.zeros((num_actions, num_states, num_states))
for a in range(num_actions):
    T[a] = transition_mats[a]

R = np.zeros((num_actions, num_states))
R[:] = mat_reward

def env_step(state, action):
    probs = T[action, state, :]
    next_state = np.random.choice(np.arange(num_states), p=probs)
    reward = R[action, state]
    return next_state, reward

alpha = 0.2
gamma = 0.8
epsilon = 0.01
num_episodes = 2000
max_steps_per_episode = 150

Q = np.zeros((num_states, num_actions))
episode_rewards = []
start_time = time.perf_counter()

for episode in range(num_episodes):
    state = np.random.randint(0, num_states)
    total_reward = 0.0

    for _ in range(max_steps_per_episode):
        if random.random() < epsilon:
            action = np.random.randint(0, num_actions)
        else:
            action = int(np.argmax(Q[state, :]))

        next_state, reward = env_step(state, action)

        best_next_action = int(np.argmax(Q[next_state, :]))
        td_target = reward + gamma * Q[next_state, best_next_action]
        Q[state, action] += alpha * (td_target - Q[state, action])

        total_reward += reward
        state = next_state

    episode_rewards.append(total_reward)

policy = np.argmax(Q, axis=1)

print("Learned Q table\n", Q, "\n")
print("Best action in each state indices", policy)
print("Best action in each state labels")
for s in range(num_states):
    print(f"  State {states_labels[s]} -> Action {actions_labels[policy[s]]}")

window_size = 50
smoothed_rewards = []
for i in range(num_episodes):
    start = max(0, i - window_size + 1)
    smoothed_rewards.append(np.mean(episode_rewards[start : i + 1]))

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Episode reward", alpha=0.5)
plt.plot(smoothed_rewards, label=f"Smoothed window {window_size}", linewidth=2)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.legend(fontsize=10)
plt.grid(visible=True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Enhanced_episode_reward.png", dpi=300, bbox_inches="tight")
plt.show()
plt.style.use("seaborn-darkgrid")  

state_ticks = [
    "S1: 0.95 < SL ≤ 1.00",
    "S2: 0.90 < SL ≤ 0.95",
    "S3: 0.80 < SL ≤ 0.90",
    "S4: 0.50 < SL ≤ 0.80",
    "S5: 0.30 < SL ≤ 0.50",
    "S6: 0.00 ≤ SL ≤ 0.30",
]

mir_levels = [1.0, 0.7, 0.4, 0.0]
mir_m = [mir_levels[(z - 1) // 4] for z in range(1, 17)]
mir_d = [mir_levels[(z - 1) % 4] for z in range(1, 17)]

action_labels = [f"A{z}" for z in range(1, 17)]
mapping_labels = [f"M {int(m*100)}%\nD {int(d*100)}%" for m, d in zip(mir_m, mir_d)]

fig, ax = plt.subplots(figsize=(18, 14))

hm = sns.heatmap(
    Q,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=np.median(Q),
    vmin=np.min(Q),
    vmax=np.max(Q),
    xticklabels=action_labels,
    yticklabels=state_ticks,
    cbar=False,
    linewidths=0,             
    linecolor=None,
    annot_kws={"size": 16, "color": "black"},
    ax=ax,
)
    
mesh = ax.collections[0]
mesh.set_edgecolor("face")
mesh.set_linewidth(0)


ax.grid(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.tick_params(axis="x", which="both", length=0)
ax.tick_params(axis="y", which="both", length=0)

ax.set_ylabel("States (service level)", fontsize=20)

ax.set_xlabel("Actions (Minimum Inventory of Manufacturers & DCs)", fontsize=20, labelpad=20)

best_actions = np.argmax(Q, axis=1)
for y, x in enumerate(best_actions):
    ax.add_patch(Rectangle((x, y), 1, 1, fill=False, linewidth=4, edgecolor="black"))

ax.tick_params(axis="x", rotation=0, labelsize=18, pad=6)
ax.tick_params(axis="y", rotation=0, labelsize=18)

fig.subplots_adjust(bottom=0.32)

plt.savefig("Figure8_Q_table_clear_no_cbar.png", dpi=300, bbox_inches="tight")
plt.show()

policy_2d = policy.reshape(1, -1)
policy_labels = np.array([mapping_labels[a] for a in policy]).reshape(1, -1)

colors = sns.color_palette("tab10", n_colors=num_actions)
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(num_actions + 1) - 0.5, num_actions)

plt.figure(figsize=(13, 2.6))
ax2 = sns.heatmap(
    policy_2d,
    cmap=cmap,
    norm=norm,
    cbar=True,
    annot=policy_labels,
    fmt="",
    annot_kws={"size": 9, "color": "black"},
    xticklabels=state_ticks,
    yticklabels=["Optimal action"],
    linewidths=0,
    linecolor=None,
)

ax2.grid(False)
ax2.tick_params(axis="x", which="both", length=0)
ax2.tick_params(axis="y", which="both", length=0)

plt.xlabel("State by service level interval", fontsize=12)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=10)

cbar = ax2.collections[0].colorbar
cbar.set_ticks(range(num_actions))
cbar.set_ticklabels(
    [f"A{z}  M {int(m*100)}%  D {int(d*100)}%" for z, (m, d) in enumerate(zip(mir_m, mir_d), start=1)]
)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("Enhanced_learned_policy_MIR.png", dpi=300, bbox_inches="tight")
plt.show()
end_time = time.perf_counter()

training_time = end_time - start_time

print(f"\nQ-learning training time: {training_time:.4f} seconds")