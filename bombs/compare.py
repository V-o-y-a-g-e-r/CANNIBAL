bombs_dir = "bombs/results"
vig_dir = "results"
import pandas as pd
import numpy as np

global_r2_vig_results = []
global_r2_bombs_results = []

global_mae_vig_results = []
global_mae_bombs_results = []

for n_bands in range(5, 161, 5):
    mae_bombs_results, mae_vig_results = [], []
    for i in range(5):
        b = pd.read_csv(f"{bombs_dir}/n_clusters-{n_bands},test_size-0.2,random_state-{i}.csv")
        v = pd.read_csv(f"{vig_dir}/n_clusters-{n_bands},test_size-0.2,random_state-{i}.csv")
        mae_bombs_results.append(b["mae_mean"].item())
        mae_vig_results.append(v["mae_mean"].item())
    r2_bombs_results, r2_vig_results = [], []
    global_mae_bombs_results.append(np.mean(mae_bombs_results).item())
    global_mae_vig_results.append(np.mean(mae_vig_results).item())
    metric_name = "r2_mean"
    for i in range(5):
        b = pd.read_csv(f"{bombs_dir}/n_clusters-{n_bands},test_size-0.2,random_state-{i}.csv")
        v = pd.read_csv(f"{vig_dir}/n_clusters-{n_bands},test_size-0.2,random_state-{i}.csv")
        r2_bombs_results.append(b["r2_mean"].item())
        r2_vig_results.append(v["r2_mean"].item())
    global_r2_bombs_results.append(np.mean(r2_bombs_results).item())
    global_r2_vig_results.append(np.mean(r2_vig_results).item())


import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Increase size and quality
fig, ax = plt.subplots(figsize=(10, 6), dpi=1000)

# Create the line plot
ax.plot(list(range(5, 161, 5)), global_r2_bombs_results, label=r'$BOMBS$')
ax.plot(list(range(5, 161, 5)), global_r2_vig_results, label=r'$VIG-Based$')

# Add labels and legend
ax.set_xlabel('Band index')
ax.set_ylabel('R2 Score')
ax.legend()

# Save the plot to a file (optional)
plt.savefig('line_plot.png', bbox_inches='tight')

# Show the plot
plt.show()
