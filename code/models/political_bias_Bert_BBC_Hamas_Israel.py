#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Israel-Gaza War Political Bias Pipeline:
- Classifies all articles with Political-BERT
- Generates overall bias bar chart, war split grouped bar, time trend line plot, and pie charts
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ─── Setup ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "output_results")
INPUT_FILE = os.path.join(BASE_DIR, "bbc_israel_2020_2024_advanced_filtered.csv")
FILTERED_FILE = os.path.join(PLOT_DIR, "bbc_israel_bias_labeled.csv")
os.makedirs(PLOT_DIR, exist_ok=True)

WAR_CUTOFF = pd.Timestamp("2023-10-07")  # Gaza war date
def split_text_into_chunks(text, tokenizer, max_tokens=512, stride=256):
    input_ids = tokenizer.encode(text, truncation=False)
    chunks = []

    for i in range(0, len(input_ids), stride):
        chunk_ids = input_ids[i:i + max_tokens]
        if not chunk_ids:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

        if i + max_tokens >= len(input_ids):
            break

    return chunks
# ─── 1. Classifier ───────────────────────────────────────────────────────────────
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import pandas as pd

def run_bias_classifier(df: pd.DataFrame, model_name="bucketresearch/politicalBiasBERT", batch_size: int = 8, device: int = 0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    preds = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
        text = row.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            preds.append("error")
            continue

        # 分段处理全文
        chunks = split_text_into_chunks(text, tokenizer, max_tokens=512, stride=256)
        if not chunks:
            preds.append("error")
            continue

        chunk_preds = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                outputs = clf(batch, batch_size=batch_size, truncation=True, max_length=512)
                labels = [o["label"] for o in outputs]
                chunk_preds.extend(labels)
            except Exception as e:
                print(f"⚠️ Error at article {idx}, chunk batch {i}: {e}")
                chunk_preds.extend(["error"] * len(batch))

        # 多数投票策略（忽略 error）
        filtered = [label for label in chunk_preds if label != "error"]
        if filtered:
            final_label = Counter(filtered).most_common(1)[0][0]
        else:
            final_label = "error"

        preds.append(final_label)

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
    plt.savefig(os.path.join(out_dir, "overall_bar_1007split.png"), dpi=300)
    plt.close()

def plot_pre_post_war_bias(df: pd.DataFrame, out_dir: str):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()]
    df["war_period"] = df["datetime"].apply(
        lambda x: "Before Oct 2023" if x < WAR_CUTOFF else "After Oct 2023"
    )
    grouped = df.groupby(["predicted_label", "war_period"]).size().unstack(fill_value=0)
    grouped = grouped.loc[[l for l in ["LEFT", "CENTER", "RIGHT"] if l in grouped.index]]

    grouped.plot(kind="bar", figsize=(8, 5), width=0.7, edgecolor='black',
                 color=["#1f77b4", "#ff7f0e"])
    plt.title("Bias Distribution Before vs After Gaza War", fontsize=14, weight="bold")
    plt.xlabel("Predicted Political Bias")
    plt.ylabel("Article Count")
    plt.xticks(rotation=0)
    plt.legend(title="Period", loc="upper right", frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bias_pre_post_1007split.png"), dpi=300)
    plt.close()

def plot_time_trend_with_split(df: pd.DataFrame, out_dir: str):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()]
    df["month"] = df["datetime"].dt.to_period("M")
    grouped = df.groupby(["month", "predicted_label"]).size().unstack(fill_value=0)
    grouped.index = grouped.index.astype(str)

    plt.figure(figsize=(12, 6))
    colors = {"LEFT": "#1f77b4", "CENTER": "#ff7f0e", "RIGHT": "#d62728"}

    for label in grouped.columns:
        plt.plot(grouped.index, grouped[label], label=label, marker="o", linewidth=2.5, color=colors.get(label, "gray"))

    plt.axvline("2023-10", linestyle="--", color="black", linewidth=1)
    plt.text("2023-10", plt.ylim()[1]*0.95, "Oct 2023", rotation=90, verticalalignment='top', fontsize=10, color="black")

    plt.title("Political Bias Trend Over Time (Gaza War Split)", fontsize=14, weight="bold")
    plt.xlabel("Month")
    plt.ylabel("Article Count")
    plt.grid(True, linestyle="--", alpha=0.5)

    xticks = grouped.index.tolist()
    interval = 3
    xtick_labels = [label if i % interval == 0 else "" for i, label in enumerate(xticks)]
    plt.xticks(ticks=range(len(xticks)), labels=xtick_labels, rotation=45, ha='right', fontsize=9)
    plt.subplots_adjust(bottom=0.18)

    plt.legend(title="Bias", loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bias_time_trend_1007split.png"), dpi=300)
    plt.close()

def plot_war_period_pie_chart(df: pd.DataFrame, out_dir: str):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()]
    df["war_period"] = df["datetime"].apply(
        lambda x: "Before Oct 2023" if x < WAR_CUTOFF else "After Oct 2023"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = {"LEFT": "#1f77b4", "CENTER": "#ff7f0e", "RIGHT": "#d62728"}

    for i, period in enumerate(["Before Oct 2023", "After Oct 2023"]):
        subset = df[df["war_period"] == period]
        label_counts = subset["predicted_label"].value_counts().reindex(["LEFT", "CENTER", "RIGHT"]).fillna(0)

        axes[i].pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=140,
                    colors=[colors.get(k, "gray") for k in label_counts.index])
        axes[i].set_title(f"{period}", fontsize=12, weight="bold")

    plt.suptitle("Bias Proportions Before vs After Gaza War", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bias_war_period_pie_1007split.png"), dpi=300)
    plt.close()

# ─── 3. Main ─────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_FILE)
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["content"].fillna("")
    df["datetime"] = pd.to_datetime(df["published_date"], errors="coerce").dt.tz_localize(None)

    if "text" not in df.columns or df["text"].isna().all():
        raise ValueError("Missing valid 'text' column after combining.")

    print(f"✅ Total articles to classify: {len(df):,}")
    df_labeled = run_bias_classifier(df, batch_size=16, device=0)
    df_labeled.to_csv(FILTERED_FILE, index=False, encoding="utf-8")
    print(f"💾 Saved predictions to: {FILTERED_FILE}")

    plot_overall_distribution(df_labeled, PLOT_DIR)
    plot_pre_post_war_bias(df_labeled, PLOT_DIR)
    plot_time_trend_with_split(df_labeled, PLOT_DIR)
    plot_war_period_pie_chart(df_labeled, PLOT_DIR)

    print("🎉 All visualizations saved in:", PLOT_DIR)

if __name__ == "__main__":
    main()
