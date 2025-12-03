# make_bias_prepost_plots.py
# 作用：对四个文件 & 模型（自动识别）生成 “战前 vs 战中” 并排柱状图（BBC | Guardian）
# 依赖：pip install pandas matplotlib

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 你的四个文件路径 =========
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

# 类别
CATEGORIES = ["Left", "Centre", "Right"]

# 图表字体
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 14
})

# 可能的日期列
DATE_CANDS = ["published_date", "time_std", "time", "date", "datetime", "pub_time", "pub_date"]


def load_csv_with_date(path: str) -> pd.DataFrame:
    """读取CSV并识别日期列为 published_date（若已有则直接解析）"""
    df = pd.read_csv(path, dtype="string", low_memory=False)

    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    else:
        col_found = None
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

    df = df.dropna(subset=["published_date"])
    return df


def normalize_label(x) -> Optional[str]:
    """规范到 Left/Centre/Right；无法识别返回 None"""
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
    自动发现各模型对应的 label 列；
    返回 { 'BERT': 'predicted_label_bert', 'DeepSeek': 'xxx', ... }
    规则：列名包含 'label' 且包含模型关键词（bert/deepseek/gemini/claude）
    """
    model_map = {}
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


def get_pre_post_proportions(df: pd.DataFrame, label_col: str, cutoff: pd.Timestamp) -> Tuple[List[float], List[float]]:
    """按 cutoff 分割，返回（pre_list, post_list），顺序为 CATEGORIES"""
    sub = df[[label_col, "published_date"]].copy()
    sub["bias_norm"] = sub[label_col].map(normalize_label)
    sub = sub.dropna(subset=["bias_norm", "published_date"])

    pre = sub[sub["published_date"] < cutoff]["bias_norm"].value_counts(normalize=True).to_dict()
    post = sub[sub["published_date"] >= cutoff]["bias_norm"].value_counts(normalize=True).to_dict()

    out_pre = [float(pre.get(cat, 0.0)) for cat in CATEGORIES]
    out_post = [float(post.get(cat, 0.0)) for cat in CATEGORIES]
    return out_pre, out_post


# ========= 颜色：两套深色，增强对比 =========
# Pre-war（深色组1）
PRE_COLORS = {
    "Left":   "#8B0000",  # 深红
    "Centre": "#B8860B",  # 深金棕
    "Right":  "#0B3D91",  # 深蓝
}
# During-war（深色组2）
POST_COLORS = {
    "Left":   "#B2182B",  # 鲜红
    "Centre": "#DAA520",  # 金色
    "Right":  "#2166AC",  # 亮蓝
}


def plot_pre_post(ax, pre_values: List[float], post_values: List[float], title: str):
    """并排柱：Pre-war 与 During-war 使用两套深色（同色系但不同色调）"""
    x = np.arange(len(CATEGORIES))
    width = 0.35

    pre_cols = [PRE_COLORS[c] for c in CATEGORIES]
    post_cols = [POST_COLORS[c] for c in CATEGORIES]

    bars1 = ax.bar(x - width/2, pre_values,  width, color=pre_cols,  label="Pre-war")
    bars2 = ax.bar(x + width/2, post_values, width, color=post_cols, label="During-war")

    # 百分比标签
    for bar_pre, bar_post in zip(bars1, bars2):
        ax.text(bar_pre.get_x() + bar_pre.get_width()/2,
                bar_pre.get_height() + 0.01,
                f"{bar_pre.get_height():.1%}",
                ha='center', va='bottom', fontsize=12, weight="bold")
        ax.text(bar_post.get_x() + bar_post.get_width()/2,
                bar_post.get_height() + 0.01,
                f"{bar_post.get_height():.1%}",
                ha='center', va='bottom', fontsize=12, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=14)
    ax.set_title(title, weight="bold", fontsize=18)
    ax.set_ylabel("Proportion of Articles", fontsize=16)
    ax.spines[['top', 'right']].set_visible(False)
    ymax = max(max(pre_values or [0]), max(post_values or [0])) * 1.25
    ax.set_ylim(0, min(1.0, max(0.15, ymax)))
    ax.legend(frameon=False)


def make_one_model_two_plots(model: str,
                             df_bbc_ip: pd.DataFrame, df_gua_ip: pd.DataFrame,
                             df_bbc_ru: pd.DataFrame, df_gua_ru: pd.DataFrame,
                             col_bbc_ip: str, col_gua_ip: str,
                             col_bbc_ru: str, col_gua_ru: str):
    """同一模型输出两张图（IP 与 RU）"""
    # 以色列–哈马斯
    bbc_pre_ip, bbc_post_ip = get_pre_post_proportions(df_bbc_ip, col_bbc_ip, WAR_START["IP"])
    gua_pre_ip, gua_post_ip = get_pre_post_proportions(df_gua_ip, col_gua_ip, WAR_START["IP"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plot_pre_post(axes[0], bbc_pre_ip, bbc_post_ip, f"{model} – BBC (Israel–Hamas)")
    plot_pre_post(axes[1], gua_pre_ip, gua_post_ip, f"{model} – Guardian (Israel–Hamas)")
    plt.tight_layout()
    out_path = OUT_DIR / f"{model}_IP_bbc_guardian_bias_pre_post.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print("保存：", out_path)

    # 俄乌
    bbc_pre_ru, bbc_post_ru = get_pre_post_proportions(df_bbc_ru, col_bbc_ru, WAR_START["RU"])
    gua_pre_ru, gua_post_ru = get_pre_post_proportions(df_gua_ru, col_gua_ru, WAR_START["RU"])

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plot_pre_post(axes2[0], bbc_pre_ru, bbc_post_ru, f"{model} – BBC (Russia–Ukraine)")
    plot_pre_post(axes2[1], gua_pre_ru, gua_post_ru, f"{model} – Guardian (Russia–Ukraine)")
    plt.tight_layout()
    out_path2 = OUT_DIR / f"{model}_RU_bbc_guardian_bias_pre_post.png"
    plt.savefig(out_path2, dpi=300)
    plt.close(fig2)
    print("保存：", out_path2)


def main():
    # 载入四个文件并解析日期
    df_bbc_ru = load_csv_with_date(BBC_RU)
    df_gua_ru = load_csv_with_date(GUA_RU)
    df_bbc_ip = load_csv_with_date(BBC_IP)
    df_gua_ip = load_csv_with_date(GUA_IP)

    # 识别各自模型列
    bbc_models_ru  = detect_model_label_cols(df_bbc_ru)
    gua_models_ru  = detect_model_label_cols(df_gua_ru)
    bbc_models_ip  = detect_model_label_cols(df_bbc_ip)
    gua_models_ip  = detect_model_label_cols(df_gua_ip)

    # 共同模型（BBC 与 Guardian 都有）
    common_models_ip = sorted(set(bbc_models_ip) & set(gua_models_ip))
    common_models_ru = sorted(set(bbc_models_ru) & set(gua_models_ru))
    common_models = sorted(set(common_models_ip) & set(common_models_ru))

    if not common_models:
        print("⚠️ 未找到在四个文件中都存在的模型列。")
        print("BBC_IP:", bbc_models_ip)
        print("GUA_IP:", gua_models_ip)
        print("BBC_RU:", bbc_models_ru)
        print("GUA_RU:", gua_models_ru)
        return

    # 逐模型输出两张图
    for model in common_models:
        make_one_model_two_plots(
            model=model,
            df_bbc_ip=df_bbc_ip, df_gua_ip=df_gua_ip,
            df_bbc_ru=df_bbc_ru, df_gua_ru=df_gua_ru,
            col_bbc_ip=bbc_models_ip[model], col_gua_ip=gua_models_ip[model],
            col_bbc_ru=bbc_models_ru[model], col_gua_ru=gua_models_ru[model]
        )


if __name__ == "__main__":
    main()
