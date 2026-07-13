# make_lr_diff_panels_vstack_academic.py
# -*- coding: utf-8 -*-
#
# Academic-styled figure (original visual style, metric switched to Right − Left):
# - One figure per conflict (RU/IP)
# - 3 vertically stacked subplots: BERT / DeepSeek / Gemini (if present)
# - y-axis: months; x-axis: Right − Left (proportion)
# - Two series per subplot: BBC vs Guardian
# - Styling: serif font, subdued grid; BBC solid+circle, Guardian dashed+square
# - Symmetric x-limits across subplots; vertical 0-line dashed; war-start line dot-dashed
#
# Requirements: keep the four CSVs in the same directory or edit the paths below.
# Dependencies: pandas, matplotlib, numpy

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= File paths =================
bbc_ru_file = "bbc_ru_bias_merge_result_filtered_with_time.csv"
guardian_ru_file = "guardian_ur2_models_merged_http_filtered_cleaned.csv"
bbc_ip_file = "bbc_bias_ip_merge_result_filtered_with_time.csv"
guardian_ip_file = "guardian_ip_models_merged_http_filtered_cleaned.csv"

plot_dir = Path("./bias_trend_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# Cutoff months (draw horizontal reference line at these months if present)
CUTOFF_MONTH = {"RU": "2022-02", "IP": "2023-10"}

# Target models to show (will be skipped if not found in both datasets)
TARGET_MODELS = ["BERT", "DeepSeek", "Gemini"]

# Controls
Y_TICK_EVERY = 4             # show one tick every 2 months
FIG_HEIGHT_PER_SUBPLOT = 5.5 # total height ~= 3 * this
MARKER_SIZE = 4
LINE_WIDTH = 2.2

# Academic styling (same as your first version)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.dpi": 500,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Colors & linestyles (match the original visual style)
STYLE_BBC = dict(color="#1b3a5a", linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, linestyle="-")
STYLE_GDN = dict(color="#b22222", linewidth=LINE_WIDTH, marker="s", markersize=MARKER_SIZE, linestyle="--")

# Candidate date columns
DATE_CANDIDATES = ["time_std", "time", "date", "published_date", "datetime", "pub_time", "pub_date"]


# ================= Utilities =================
def load_csv_and_add_time(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string", low_memory=False)

    date_col = None
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand in lower_cols:
            date_col = lower_cols[cand]
            break
    if date_col is None:
        for c in df.columns:
            lc = c.lower()
            if any(tok in lc for tok in ["date", "time", "publish", "datetime", "created", "updated"]):
                date_col = c
                break
    if date_col is None:
        raise ValueError(f"{path} has no recognizable date column.")
    df["time_std"] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.dropna(subset=["time_std"])
    df["month"] = df["time_std"].dt.strftime("%Y-%m")
    return df


def normalize_label(label) -> Optional[str]:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    s = str(label).strip().lower()
    if "left" in s:
        return "Left"
    if "centre" in s or "center" in s:
        return "Centre"
    if "right" in s:
        return "Right"
    return None


def detect_model_label_cols(df: pd.DataFrame) -> Dict[str, str]:
    model_map: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if "label" not in lc:
            continue
        if "url" in lc or "title" in lc:
            continue
        if "bert" in lc:
            model_map["BERT"] = c
        elif "deepseek" in lc:
            model_map["DeepSeek"] = c
        elif "gemini" in lc:
            model_map["Gemini"] = c
        elif "claude" in lc:
            model_map["Claude"] = c
    return model_map


def build_long_df_for_model(bbc_df: pd.DataFrame, gdn_df: pd.DataFrame,
                            label_col_bbc: str, label_col_gdn: str) -> pd.DataFrame:
    b = bbc_df[["month", label_col_bbc]].copy()
    b["source"] = "BBC"
    b["label"] = b[label_col_bbc].map(normalize_label)

    g = gdn_df[["month", label_col_gdn]].copy()
    g["source"] = "Guardian"
    g["label"] = g[label_col_gdn].map(normalize_label)

    df = pd.concat([b[["source","month","label"]], g[["source","month","label"]]], ignore_index=True)
    df = df.dropna(subset=["month","label"])
    return df


def compute_monthly_rl_diff(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    For each (source, month):
        rl_diff = P(Right) - P(Left)
    Return: columns [source, month, rl_diff]
    """
    counts = df_long.groupby(["source","month","label"]).size().reset_index(name="count")
    totals = counts.groupby(["source","month"])["count"].transform("sum")
    counts["prop"] = counts["count"] / totals
    piv = counts.pivot_table(index=["source","month"], columns="label", values="prop", fill_value=0).reset_index()
    right = piv["Right"] if "Right" in piv.columns else 0
    left  = piv["Left"]  if "Left"  in piv.columns else 0
    piv["rl_diff"] = right - left
    return piv[["source","month","rl_diff"]].copy()


def draw_three_model_panels(conflict_key: str,
                            per_model_data: Dict[str, pd.DataFrame],
                            title_suffix: str,
                            out_path: Path):
    """
    - per_model_data: {model_name -> DataFrame[source, month, rl_diff]}
    - y-axis: months (ascending), x-axis: rl_diff
    - Keep original visual style (markers on, BBC solid / Guardian dashed)
    """
    # Compute global symmetric x-limit across all available models
    all_vals = []
    for df in per_model_data.values():
        if df is not None and not df.empty:
            all_vals.append(df["rl_diff"].to_numpy())
    vmax = float(np.nanmax(np.abs(np.concatenate(all_vals)))) if all_vals else 0.0
    vmax = max(0.2, round(vmax + 0.05, 2))

    fig, axes = plt.subplots(3, 1, figsize=(10, 3*FIG_HEIGHT_PER_SUBPLOT), sharex=True)

    ordered_models = ["BERT", "DeepSeek", "Gemini"]
    for i, model in enumerate(ordered_models):
        ax = axes[i]
        df = per_model_data.get(model, None)
        if df is None or df.empty:
            ax.set_title(f"{model} (no data)")
            ax.axis("off")
            continue

        months = sorted(df["month"].unique())
        y = np.arange(len(months))

        # Series: BBC & Guardian
        sub_bbc = df[df["source"] == "BBC"].set_index("month").reindex(months)
        sub_gdn = df[df["source"] == "Guardian"].set_index("month").reindex(months)

        ax.plot(sub_bbc["rl_diff"].to_numpy(), y, label="BBC", **STYLE_BBC)
        ax.plot(sub_gdn["rl_diff"].to_numpy(), y, label="Guardian", **STYLE_GDN)

        # Reference lines (as in the original style)
        ax.axvline(0, linestyle="--", color="black", linewidth=1.2)           # neutrality (dashed)
        cutoff = CUTOFF_MONTH.get(conflict_key)
        if cutoff in months:
            idx = months.index(cutoff)
            ax.axhline(idx, linestyle=(0, (4, 3)), color="#777777", linewidth=1.0)  # war start (dot-dash)

        # Axes & grids
        ax.set_title(f"{model}")
        ax.set_ylabel("Month")

        TICK_INTERVAL = 4
        tick_idx = np.arange(0, len(months), TICK_INTERVAL)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([months[j] for j in tick_idx])

        # subtle vertical grid only
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.grid(axis="y", visible=False)

        ax.set_xlim(-vmax, vmax)

        if i == 0:
            ax.legend(loc="upper left", frameon=False)

    axes[-1].set_xlabel("Right − Left (proportion)")
    fig.suptitle(f"BBC vs Guardian  by Model ({title_suffix})", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def make_figure_for_conflict(conflict_key: str, bbc_path: str, gdn_path: str, title_suffix: str):
    bbc_df = load_csv_and_add_time(bbc_path)
    gdn_df = load_csv_and_add_time(gdn_path)

    bbc_models = detect_model_label_cols(bbc_df)
    gdn_models = detect_model_label_cols(gdn_df)

    common_models = [m for m in TARGET_MODELS if (m in bbc_models and m in gdn_models)]
    if not common_models:
        print(f"⚠️ No common target models ({TARGET_MODELS}) for {conflict_key}. BBC={bbc_models}, GDN={gdn_models}")
        return

    per_model_data: Dict[str, pd.DataFrame] = {}
    for m in common_models:
        df_long = build_long_df_for_model(bbc_df, gdn_df, bbc_models[m], gdn_models[m])
        if df_long.empty:
            continue
        per_model_data[m] = compute_monthly_rl_diff(df_long)

    out_path = plot_dir / f"RLdiff_{conflict_key}_BERT_DeepSeek_Gemini_VSTACK_ACAD.png"
    draw_three_model_panels(conflict_key, per_model_data, title_suffix, out_path)


def main():
    make_figure_for_conflict("RU", bbc_ru_file, guardian_ru_file, "Russia–Ukraine")
    make_figure_for_conflict("IP", bbc_ip_file, guardian_ip_file, "Israel–Hamas")


if __name__ == "__main__":
    main()
