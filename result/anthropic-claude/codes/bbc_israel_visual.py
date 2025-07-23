import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ========== Configuration ==========
INPUT_FILE = "claude_bbc_israel_2020_2024_with_leaning.csv"

# Color-blind friendly colors
color_map = {
    "Left": "#B2182B",    # Deep red
    "Centre": "#FDD835",  # Yellow/golden
    "Right": "#2166AC"    # Deep blue
}

# ========== Load and Clean Data ==========
df = pd.read_csv(INPUT_FILE)

# Parse date column
df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

# Standardize 'leaning' values
df["leaning"] = df["leaning"].str.strip().str.capitalize()
df["leaning"] = df["leaning"].replace({"Center": "Centre"})

# Drop rows with missing data
df = df.dropna(subset=["published_date", "leaning"])

# Create month column
df["month"] = df["published_date"].dt.to_period("M").dt.to_timestamp()

# Group by month and leaning
monthly_counts = df.groupby(["month", "leaning"]).size().unstack(fill_value=0)
monthly_counts = monthly_counts.sort_index()

# ========== Line Chart ==========
plt.figure(figsize=(14, 6))

for category in ["Left", "Centre", "Right"]:
    if category in monthly_counts.columns:
        plt.plot(
            monthly_counts.index,
            monthly_counts[category],
            color=color_map[category],
            marker="o",
            label=category
        )

# Add vertical line for 7 Oct 2023 (Israel-Hamas war start)
plt.axvline(pd.to_datetime("2023-10-07"), color="black", linestyle="--", linewidth=1)

# Style
plt.title("Claude - BBC Bias Trend (Hamas-Israel)", fontsize=14, weight="bold")
plt.xlabel("Month")
plt.ylabel("Number of Articles")
plt.legend(title="", loc="best")
plt.grid(False)
plt.xticks(rotation=45, ha="right")
plt.gca().spines[['top', 'right']].set_visible(False)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# Export
plt.tight_layout()
plt.savefig("claude_bbc_monthly_bias_trend_hamas_israel.png", dpi=300)
plt.show()

# ========== Pie Chart ==========
total_counts = df["leaning"].value_counts().to_dict()

for cat in ["Left", "Centre", "Right"]:
    if cat not in total_counts:
        total_counts[cat] = 0

labels = ["Left", "Centre", "Right"]
sizes = [total_counts["Left"], total_counts["Centre"], total_counts["Right"]]
colors = [color_map[label] for label in labels]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
plt.title("Claude - BBC Overall Bias Proportion (Hamas-Israel)", fontsize=14, weight="bold")
plt.axis("equal")
plt.tight_layout()
plt.savefig("claude_bbc_pie_bias_distribution_hamas_israel.png", dpi=300)
plt.show()

# ========== Bar Chart ==========
plt.figure(figsize=(8, 6))
plt.bar(labels, sizes, color=colors)
plt.title("Claude - BBC Overall Bias Count (Hamas-Israel)", fontsize=14, weight="bold")
plt.xlabel("Bias Category")
plt.ylabel("Number of Articles")

# Add value labels
for i, v in enumerate(sizes):
    plt.text(i, v + max(sizes)*0.01, str(v), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("claude_bbc_bar_bias_distribution_hamas_israel.png", dpi=300)
plt.show()

# ========== Pre-war vs During-war Bar Chart ==========
war_start = pd.to_datetime("2023-10-07")
pre_war = df[df["published_date"] < war_start]
post_war = df[df["published_date"] >= war_start]

pre_counts = pre_war["leaning"].value_counts(normalize=True).to_dict()
post_counts = post_war["leaning"].value_counts(normalize=True).to_dict()

for cat in ["Left", "Centre", "Right"]:
    pre_counts.setdefault(cat, 0)
    post_counts.setdefault(cat, 0)

categories = ["Left", "Centre", "Right"]
pre_values = [pre_counts[cat] for cat in categories]
post_values = [post_counts[cat] for cat in categories]
colors = [color_map[cat] for cat in categories]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8, 6))
bars1 = plt.bar(x - width/2, pre_values, width, color=colors, alpha=0.4)
bars2 = plt.bar(x + width/2, post_values, width, color=colors, alpha=1.0)

# Add data labels
for i, (bar_pre, bar_post) in enumerate(zip(bars1, bars2)):
    plt.text(bar_pre.get_x() + bar_pre.get_width()/2, bar_pre.get_height() + 0.01,
             f"Pre-war\n{bar_pre.get_height():.1%}", ha='center', va='bottom', fontsize=9)
    plt.text(bar_post.get_x() + bar_post.get_width()/2, bar_post.get_height() + 0.01,
             f"During-war\n{bar_post.get_height():.1%}", ha='center', va='bottom', fontsize=9)

plt.xticks(ticks=x, labels=categories, fontsize=11)
plt.title("Claude - BBC Bias Distribution Pre/During War (Hamas-Israel)", fontsize=14, weight="bold")
plt.ylabel("Proportion of Articles")
plt.ylim(0, max(max(pre_values), max(post_values)) * 1.25)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("claude_bbc_bias_comparison_pre_post_hamas_israel.png", dpi=300)
plt.show()