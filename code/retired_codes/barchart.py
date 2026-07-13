# make_bias_prepost_periods_xaxis_thick.py
# 两张图（IP / RU），以“时期（媒体×战前战中）”为横坐标。
# 颜色=类别（Left/Centre/Right）；hatch=模型（BERT/DeepSeek/Gemini）。
# 柱更粗、文字更大、柱顶显示百分比。

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 数据文件 =========
BBC_RU = "bbc_ru_bias_merge_result_filtered_with_time.csv"
GUA_RU = "guardian_ur2_models_merged_http_filtered_cleaned.csv"
BBC_IP = "bbc_bias_ip_merge_result_filtered_with_time.csv"
GUA_IP = "guardian_ip_models_merged_http_filtered_cleaned.csv"

# 输出目录
OUT_DIR = Path("./bias_prepost_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 分界日期
WAR_START = {
    "IP": pd.to_datetime("2023-10-07"),  # 以色列–哈马斯
    "RU": pd.to_datetime("2022-02-24"),  # 俄乌战争
}

# 组（时期）顺序
GROUPS = [
    ("BBC", "Pre-war"),
    ("BBC", "During-war"),
    ("Guardian", "Pre-war"),
    ("Guardian", "During-war"),
]

# 类别与颜色
CATEGORIES = ["Left", "Centre", "Right"]
CAT_COLORS = {
    "Left":   "#B2182B",  # 深红
    "Centre": "#2166AC",  # 深蓝
    "Right":  "#FDD835",  # 亮黄
}

# 模型与样式
MODEL_ORDER = ["BERT", "DeepSeek", "Gemini"]
MODEL_HATCH = {"BERT": "", "DeepSeek": "///", "Gemini": "xxx"}
MODEL_EDGE = {"BERT": "#1b1b1b", "DeepSeek": "#1b1b1b", "Gemini": "#1b1b1b"}

# 全局字体（更大）
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 26,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

DATE_CANDS = ["published_date", "time_std", "time", "date", "datetime", "pub_time", "pub_date"]


def load_csv_with_date(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string", low_memory=False)
    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    else:
        col_found: Optional[str] = None
        lower = {c.lower(): c for c in df.columns}
        for cand in DATE_CANDS:
            if cand in lower:
                col_found = lower[cand]
                break
        if col_found is None:
            for c in df.columns:
                lc = c.lower()
                if any(tok in lc for tok in ["date", "time", "publish", "datetime", "created", "updated"]):
                    col_found = c
                    break
        if col_found is None:
            raise ValueError(f"{path} 未找到日期列，请检查。")
        df["published_date"] = pd.to_datetime(df[col_found], errors="coerce")
    return df.dropna(subset=["published_date"]).copy()


def normalize_label(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if "left" in s:
        return "Left"
    if "centre" in s or "center" in s:
        return "Centre"
    if "right" in s:
        return "Right"
    return None


def detect_model_label_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    识别各模型的label列；返回 {'BERT': '...', 'DeepSeek': '...', 'Gemini': '...'}
    列名包含 label/bias 且包含模型关键词（bert/deepseek/gemini）
    """
    model_map: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if ("label" not in lc) and ("bias" not in lc):
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
            pass
    return model_map


def get_pre_post_proportions(df: pd.DataFrame, label_col: str, cutoff: pd.Timestamp) -> Tuple[List[float], List[float]]:
    """
    显式计算：
    - pre_list: 以战前总样本数为分母的 [Left, Centre, Right] 占比
    - post_list: 以战后总样本数为分母的 [Left, Centre, Right] 占比
    """
    sub = df[[label_col, "published_date"]].copy()
    sub["bias_norm"] = sub[label_col].map(normalize_label)
    sub = sub.dropna(subset=["bias_norm", "published_date"])

    pre_df = sub[sub["published_date"] < cutoff]
    post_df = sub[sub["published_date"] >= cutoff]

    pre_total = len(pre_df)
    post_total = len(post_df)

    pre_counts = pre_df["bias_norm"].value_counts().to_dict()
    post_counts = post_df["bias_norm"].value_counts().to_dict()

    # 分母分别用各自时期总数（若为 0，比例设为 0.0）
    out_pre = [(pre_counts.get(cat, 0) / pre_total) if pre_total > 0 else 0.0 for cat in CATEGORIES]
    out_post = [(post_counts.get(cat, 0) / post_total) if post_total > 0 else 0.0 for cat in CATEGORIES]
    return out_pre, out_post


def plot_periods_xaxis(
    ax,
    # 每个模型在四个时期的比例：{model: {'BBC': {'Pre-war':[L,C,R], 'During-war':[L,C,R]}, 'Guardian': {...}}}
    dist_by_model: Dict[str, Dict[str, Dict[str, List[float]]]],
    title: str
):
    """
    在同一张子图上绘制四个“时期”组：BBC-Pre / BBC-During / Guardian-Pre / Guardian-During。
    每个时期组内，先 Left/Centre/Right；每个类别下并排三根柱（BERT/DeepSeek/Gemini）。
    """
    n_groups = len(GROUPS)          # 4
    n_cats = len(CATEGORIES)        # 3
    n_models = len(MODEL_ORDER)     # 3

    # 粗柱设置：组间距与宽度
    model_bar_width = 0.22          # 单根柱宽（更粗）
    cat_gap = 0.06                  # 同一时期内，不同类别之间的空隙
    model_gap = 0.02                # 同一类别内，3 个模型之间微小空隙

    # 一个“类别块”的宽度
    cat_block = n_models * model_bar_width + (n_models - 1) * model_gap
    # 一个“时期组”的宽度
    group_block = n_cats * cat_block + (n_cats - 1) * cat_gap

    # 组中心
    x_centers = np.arange(n_groups) * (group_block + 0.6)

    fig_ymax = 0.0

    for gi, (outlet, phase) in enumerate(GROUPS):
        group_start = x_centers[gi] - group_block / 2
        for ci, cat in enumerate(CATEGORIES):
            cat_start = group_start + ci * (cat_block + cat_gap)
            for mi, model in enumerate(MODEL_ORDER):
                vals = dist_by_model[model][outlet][phase]  # [L, C, R]
                v = vals[ci]
                xpos = cat_start + mi * (model_bar_width + model_gap)
                bar = ax.bar(
                    xpos, v,
                    width=model_bar_width,
                    color=CAT_COLORS[cat],
                    edgecolor=MODEL_EDGE[model],
                    hatch=MODEL_HATCH[model],
                    linewidth=1.0
                )
                if v > 0:
                    ax.text(xpos, v + 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=16, weight="bold")
                fig_ymax = max(fig_ymax, v)

    # 轴外观
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"{o}\n{p}" for (o, p) in GROUPS])
    ax.set_ylabel("Proportion of Articles")
    ax.set_title(title, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(0, min(1.0, max(0.2, fig_ymax * 1.28)))

    # 左上角类别颜色图例；右上角模型纹理图例
    cat_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CAT_COLORS[c]) for c in CATEGORIES]
    leg1 = ax.legend(cat_handles, CATEGORIES, title="Category", loc="upper left", frameon=False)
    model_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#DDDDDD",
                                   edgecolor=MODEL_EDGE[m], hatch=MODEL_HATCH[m]) for m in MODEL_ORDER]
    ax.legend(model_handles, MODEL_ORDER, title="Model", loc="upper right", frameon=False)
    ax.add_artist(leg1)


def make_figure_for_conflict(
    conflict: str,
    df_bbc: pd.DataFrame, df_gua: pd.DataFrame,
    colmap_bbc: Dict[str, str], colmap_gua: Dict[str, str]
):
    cutoff = WAR_START[conflict]
    common_models = [m for m in MODEL_ORDER if (m in colmap_bbc and m in colmap_gua)]
    if not common_models:
        print(f"{conflict}: 无共同模型，跳过")
        return

    # 准备：dist_by_model[model][outlet][phase] = [L, C, R]
    dist_by_model: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for m in common_models:
        pre_bbc, post_bbc = get_pre_post_proportions(df_bbc, colmap_bbc[m], cutoff)
        pre_gua, post_gua = get_pre_post_proportions(df_gua, colmap_gua[m], cutoff)
        dist_by_model[m] = {
            "BBC": {"Pre-war": pre_bbc, "During-war": post_bbc},
            "Guardian": {"Pre-war": pre_gua, "During-war": post_gua},
        }

    # 单图（不分子图；X 轴就是 4 个时期）
    fig, ax = plt.subplots(figsize=(20, 8))
    conflict_name = "Israel–Hamas" if conflict == "IP" else "Russia–Ukraine"
    plot_periods_xaxis(ax, dist_by_model, f"All Models – {conflict_name} (Pre vs During)")

    plt.tight_layout()
    out_path = OUT_DIR / f"ALLMODELS_{conflict}_periods.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print("保存：", out_path)


def main():
    # 载入
    df_bbc_ru = load_csv_with_date(BBC_RU)
    df_gua_ru = load_csv_with_date(GUA_RU)
    df_bbc_ip = load_csv_with_date(BBC_IP)
    df_gua_ip = load_csv_with_date(GUA_IP)

    # 检测模型列
    bbc_models_ru = detect_model_label_cols(df_bbc_ru)
    gua_models_ru = detect_model_label_cols(df_gua_ru)
    bbc_models_ip = detect_model_label_cols(df_bbc_ip)
    gua_models_ip = detect_model_label_cols(df_gua_ip)

    # 画两张图
    make_figure_for_conflict("IP", df_bbc_ip, df_gua_ip, bbc_models_ip, gua_models_ip)
    make_figure_for_conflict("RU", df_bbc_ru, df_gua_ru, bbc_models_ru, gua_models_ru)


if __name__ == "__main__":
    main()
