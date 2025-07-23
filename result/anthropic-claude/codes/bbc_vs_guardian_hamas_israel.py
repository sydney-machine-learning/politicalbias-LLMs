import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== Config ==========
BBC_FILE = "claude_bbc_israel_2020_2024_with_leaning.csv"
GUARDIAN_FILE = "claude_guardian_israel_hamas_with_leaning.csv"
WAR_DATE = pd.to_datetime("2023-10-07")
group_labels = ["Left", "Centre", "Right"]
base_color = "#4C78A8"

# ========== Helper ==========
def get_leaning_proportions(df, war_date):
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df["leaning"] = df["leaning"].str.strip().str.capitalize()
    df["leaning"] = df["leaning"].replace({"Center": "Centre"})
    df = df.dropna(subset=["published_date", "leaning"])
    
    pre = df[df["published_date"] < war_date]
    post = df[df["published_date"] >= war_date]
    
    def calc_prop(subset):
        vc = subset["leaning"].value_counts(normalize=True)
        return {k: vc.get(k, 0) for k in group_labels}
    
    return calc_prop(pre), calc_prop(post)

# ========== Load Data ==========
bbc_df = pd.read_csv(BBC_FILE)
guardian_df = pd.read_csv(GUARDIAN_FILE)

bbc_pre, bbc_post = get_leaning_proportions(bbc_df, WAR_DATE)
guardian_pre, guardian_post = get_leaning_proportions(guardian_df, WAR_DATE)

# ========== Plot Preparation ==========
x = np.arange(len(group_labels))  # positions: 0, 1, 2
width = 0.18

entries = [
    ("BBC", "Pre-war", bbc_pre, 0, 0.4),
    ("BBC", "During-war", bbc_post, 1, 1.0),
    ("Guardian", "Pre-war", guardian_pre, 2, 0.4),
    ("Guardian", "During-war", guardian_post, 3, 1.0),
]

plt.figure(figsize=(10, 6))
legend_labels = {}

for media, period, dist, idx, alpha in entries:
    offset = (idx - 1.5) * width
    values = [dist[cat] for cat in group_labels]
    label = period if period not in legend_labels else None
    bars = plt.bar(x + offset, values, width, color=base_color, alpha=alpha, label=label)
    
    # Add media label above each bar
    for xi, val in zip(x + offset, values):
        plt.text(xi, val + 0.01, media, ha='center', va='bottom', fontsize=9)

    legend_labels[period] = True

# Y-axis in percentage
plt.ylabel("Proportion of Articles", fontsize=12)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(v*100)}%" for v in np.arange(0, 1.1, 0.1)])

# X-axis group labels
plt.xticks(ticks=x, labels=group_labels, fontsize=12, weight="bold")

# Title and legend
plt.title("Claude - BBC vs. Guardian Bias by Percentage Pre/During War (Hamas-Israel)", fontsize=14, weight="bold")
plt.legend(title="", loc="upper right", fontsize=9)
plt.gca().spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("bbc_guardian_grouped_bias_comparison_hamas_israel_final.png", dpi=300)
plt.show()