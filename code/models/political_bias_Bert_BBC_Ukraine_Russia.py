#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ukraine War Political Bias Pipeline:
- Classifies all articles with Political-BERT
- Produces enhanced plots: bar, grouped bar, time trend, and pie charts
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "output_results")
INPUT_FILE = os.path.join(BASE_DIR, "bbc_ukraine_2020_2024_advanced_filtered.csv")
FILTERED_FILE = os.path.join(PLOT_DIR, "bbc_ukraine_bias_labeled.csv")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── 1. Classifier ───────────────────────────────────────────────────────────────
def run_bias_classifier(df: pd.DataFrame, model_name="bucketresearch/politicalBiasBERT", batch_size=16, device=0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer,
                   truncation=True, max_length=512, device=device)

    preds = []
    texts = df["text"].tolist()
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch = texts[i:i + batch_size]
        try:
            outputs = clf(batch, batch_size=batch_size, truncation=True, max_length=512)
            preds.extend(o["label"] for o in outputs)
        except Exception as e:
            print(f"⚠️ Error at batch {i}: {e}")
            preds.extend(["error"] * len(batch))
    df = df.copy()
    df["predicted_label"] = preds
    return df

# ─── 2. Plots ────────────────────────────────────────────────────────────────────
def plot_overall_distribution(df: pd.DataFrame, out_dir: str):
    sns.set(style="whitegrid", font_scale=1.2)
    counts = df["predicted_label"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x=counts.values, y=counts.index, palette="Blues_d", edgecolor=".2")
    ax.set_title("Overall Political Bias Distribution", fontsize=14, weight="bold")
    ax.set_xlabel("Article Count")
    ax.set_ylabel("Bias Label")
    for i, v in enumerate(counts.values):
        ax.text(v + 2, i, str(v), va='center')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overall_bar.png"), dpi=300)
    plt.close()

def plot_pre_post_war_bias(df: pd.DataFrame, out_dir: str):
    sns.set(style="whitegrid", font_scale=1.2)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()]
    df["war_period"] = df["datetime"].apply(
        lambda x: "Before Feb 2022" if x < pd.Timestamp("2022-02-24") else "After Feb 2022"
    )
    grouped = df.groupby(["predicted_label", "war_period"]).size().unstack(fill_value=0)
    ordered_labels = [label for label in ["LEFT", "CENTER", "RIGHT"] if label in grouped.index]
    grouped = grouped.loc[ordered_labels]

    grouped.plot(kind="bar", figsize=(8, 5), width=0.7, edgecolor='black', linewidth=1,
                 color=["#1f77b4", "#ff7f0e"])
    plt.title("Bias Distribution Before vs After Ukraine War", fontsize=14, weight="bold")
    plt.xlabel("Predicted Political Bias")
    plt.ylabel("Article Count")
    plt.xticks(rotation=0)
    plt.legend(title="Period", loc="upper right", frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bias_pre_post_war.png"), dpi=300)
    plt.close()

def plot_time_trend_with_split(df: pd.DataFrame, out_dir: str):
    sns.set(style="whitegrid", font_scale=1.1)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()]
    df["month"] = df["datetime"].dt.to_period("M")
    grouped = df.groupby(["month", "predicted_label"]).size().unstack(fill_value=0)
    grouped.index = grouped.index.astype(str)

    plt.figure(figsize=(12, 6))
    colors = {"LEFT": "#1f77b4", "CENTER": "#ff7f0e", "RIGHT": "#d62728"}

    for label in grouped.columns:
        plt.plot(grouped.index, grouped[label], label=label,
                 marker="o", linewidth=2.5, color=colors.get(label, "gray"))

    plt.axvline("2022-02", linestyle="--", color="black", linewidth=1)
    plt.text("2022-02", plt.ylim()[1]*0.95, "Feb 2022", rotation=90,
             verticalalignment='top', fontsize=10, color="black")

    plt.title("Political Bias Trend Over Time (Ukraine War Split)", fontsize=14, weight="bold")
    plt.xlabel("Month")
    plt.ylabel("Article Count")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Improve x-axis readability
    xticks = grouped.index.tolist()
    interval = 3
    xtick_labels = [label if i % interval == 0 else "" for i, label in enumerate(xticks)]
    plt.xticks(ticks=range(len(xticks)), labels=xtick_labels, rotation=45, ha='right', fontsize=9)
    plt.subplots_adjust(bottom=0.18)

    plt.legend(title="Bias", loc="upper left", frameon=True)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "bias_time_trend_split.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_war_period_pie_chart(df: pd.DataFrame, out_dir: str):
    sns.set(style="whitegrid", font_scale=1.1)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()]
    df["war_period"] = df["datetime"].apply(
        lambda x: "Before Feb 2022" if x < pd.Timestamp("2022-02-24") else "After Feb 2022"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = {"LEFT": "#1f77b4", "CENTER": "#ff7f0e", "RIGHT": "#d62728"}

    for i, period in enumerate(["Before Feb 2022", "After Feb 2022"]):
        subset = df[df["war_period"] == period]
        label_counts = subset["predicted_label"].value_counts().sort_index()
        label_counts = label_counts.reindex(["LEFT", "CENTER", "RIGHT"]).fillna(0)

        axes[i].pie(
            label_counts,
            labels=label_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=[colors.get(k, "gray") for k in label_counts.index]
        )
        axes[i].set_title(f"{period}", fontsize=12, weight="bold")

    plt.suptitle("Political Bias Proportions: Before vs After Ukraine War", fontsize=14, weight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "bias_war_period_pie.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🥧 Saved pie chart: {out_path}")

# ─── 3. Main ─────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_FILE)

    df["text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["content"].fillna("")
    )
    df["datetime"] = pd.to_datetime(df["published_date"], errors="coerce").dt.tz_localize(None)

    if "text" not in df.columns or df["text"].isna().all():
        raise ValueError("Missing valid 'text' column after combining.")

    df_filtered = df.copy()
    print(f"✅ Total articles to classify: {len(df_filtered):,}")

    df_labeled = run_bias_classifier(df_filtered, batch_size=16, device=0)
    df_labeled.to_csv(FILTERED_FILE, index=False, encoding="utf-8")
    print(f"💾 Saved predictions to: {FILTERED_FILE}")

    plot_overall_distribution(df_labeled, PLOT_DIR)
    plot_pre_post_war_bias(df_labeled, PLOT_DIR)
    plot_time_trend_with_split(df_labeled, PLOT_DIR)
    plot_war_period_pie_chart(df_labeled, PLOT_DIR)

    print("🎉 All visualizations saved in:", PLOT_DIR)

if __name__ == "__main__":
    main()
