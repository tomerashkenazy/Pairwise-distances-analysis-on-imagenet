import json
import matplotlib.pyplot as plt

json_path = "/home/tomer_a/Documents/KDE-analysis-on-imagenet/results_imagenet_stats/nn_results.json"
with open(json_path, "r") as f:
    results = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

# -------- Exact NN --------
for norm, vals in results.items():
    k = [v[0] for v in vals]
    acc = [float(v[1]) for v in vals]
    ax1.plot(k, acc, marker='o', label=norm)

# Chance baseline: k/1000
k_vals = k
chance_exact = [kk / 1000 for kk in k_vals]
ax1.plot(k_vals, [val * 100 for val in chance_exact], "r--", linewidth=2, label="Chance (k/1000)")

ax1.set_title("Top-K NN Accuracy (Exact Class)")
ax1.set_xlabel("k")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(0, 7)
ax1.set_xticks(k_vals)
ax1.grid(True)
ax1.legend()


# -------- Superclass NN --------
for norm, vals in results.items():
    k = [v[0] for v in vals]
    sacc = [float(v[2]) for v in vals]
    ax2.plot(k, sacc, marker='o', label=norm)

# Chance baseline: k/80
chance_super = [kk / 80 for kk in k_vals]
ax2.plot(k_vals, [val * 100 for val in chance_super], "r--", linewidth=2, label="Chance (k/80)")

ax2.set_title("Top-K NN Accuracy (Superclass)")
ax2.set_xlabel("k")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0,27)
ax2.set_xticks(k_vals)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("nn_accuracy_comparison.png", dpi=200)
