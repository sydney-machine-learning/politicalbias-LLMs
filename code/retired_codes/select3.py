# make_bias_trend_tables_vertical.py
# 功能：
# - 将整段时间均分为6段（按月份连续切块），段名直接显示为“起止月份”
# - 每段分别计算 BBC 与 Guardian、各模型(BERT/DeepSeek/Gemini)的 Left/Centre/Right 平均比例与样本数 N
# - 只输出 2 个 sheet：Russia-Ukraine、Israel-Hamas（每个 sheet 同时包含 BBC 和 Guardian）
#
# 依赖：pip install pandas openpyxl
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

# ===== 四个文件路径 =====
BBC_RU = "bbc_ru_bias_merge_result_filtered_with_time.csv"
GUA_RU = "guardian_ur2_models_merged_http_filtered_cleaned.csv"
BBC_IP = "bbc_bias_ip_merge_result_filtered_with_time.csv"
GUA_IP = "guardian_ip_models_merged_http_filtered_cleaned.csv"

OUT_XLSX = Path("bias_trend_tables_vertical.xlsx")

# ===== 配置 =====
NUM_SEGMENTS = 6               # 把完整时间线均分为多少段
USE_WEIGHTED_MEAN = True       # True=加权平均（权重=该月总N），False=简单平均
BIAS_ORDER = ["Left", "Centre", "Right"]
MODELS_ORDER = ["BERT", "DeepSeek", "Gemini"]
DATE_CANDIDATES = ["time_std", "time", "date", "published_date", "datetime", "pub_time", "pub_date"]

# ================= 工具函数 =================
def load_csv_and_add_month(path: str) -> pd.DataFrame:
    """加载 CSV，识别时间列，生成标准月份 month=YYYY-MM"""
    df = pd.read_csv(path, dtype="string", low_memory=False)

    date_col = None
    lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand in lower:
            date_col = lower[cand]
            break
    if date_col is None:
        for c in df.columns:
            lc = c.lower()
            if any(tok in lc for tok in ["date", "time", "publish", "datetime", "created", "updated"]):
                date_col = c
                break
    if date_col is None:
        raise ValueError(f"{path} 未找到可识别的日期列。")

    ts = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.assign(time_std=ts).dropna(subset=["time_std"]).copy()
    df["month"] = df["time_std"].dt.strftime("%Y-%m")
    return df


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
    """自动发现模型对应的 label 列（忽略 URL/title 等非预测列）"""
    m = {}
    for c in df.columns:
        lc = c.lower()
        if "label" not in lc:
            continue
        if "url" in lc or "title" in lc:
            continue
        if "bert" in lc:
            m["BERT"] = c
        elif "deepseek" in lc:
            m["DeepSeek"] = c
        elif "gemini" in lc:
            m["Gemini"] = c
        elif "claude" in lc:
            m["Claude"] = c
    return m


def month_range(min_m: str, max_m: str) -> pd.Index:
    """生成 [min_m, max_m] 连续月份（YYYY-MM）"""
    pr = pd.period_range(min_m, max_m, freq="M")
    return pd.Index(pr.strftime("%Y-%m"))


def monthly_proportions(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    构造“月度比例表”：month, Left, Centre, Right, total
    Left/Centre/Right 为该月内该类别占比（分母为该月总数）
    """
    s = df[label_col].map(normalize_label)
    g = (
        pd.DataFrame({"month": df["month"], "label": s})
        .dropna()
        .groupby(["month", "label"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = g.groupby("month")["count"].sum().rename("total")
    wide = g.pivot(index="month", columns="label", values="count").fillna(0)

    # 确保三类列存在
    for k in BIAS_ORDER:
        if k not in wide.columns:
            wide[k] = 0

    wide = wide[BIAS_ORDER].astype(float)
    wide["total"] = totals.reindex(wide.index).fillna(0).astype(float)

    # 比例
    for k in BIAS_ORDER:
        wide[k] = wide[k] / wide["total"].where(wide["total"] > 0, other=pd.NA)

    wide = wide.reset_index()  # month -> 列
    return wide


def split_months_into_segments(all_months: pd.Index, k: int) -> List[List[str]]:
    """把完整月份序列均分成 k 段（连续且尽量均匀），返回 [[m1..], [..], ...] 共k段"""
    idxs = np.array_split(np.arange(len(all_months)), k)
    return [[all_months[int(i)] for i in seg] for seg in idxs]


def segment_weighted_mean(wide: pd.DataFrame, months: List[str]) -> Tuple[Dict[str, float], int]:
    """
    对给定月份列表做均值：
    - 若 USE_WEIGHTED_MEAN=True：按当月 total 加权平均；否则简单平均
    返回（{Left/Centre/Right 均值}, 段内总 N）
    """
    df = wide[wide["month"].isin(months)].copy()
    if df.empty:
        return {k: float("nan") for k in BIAS_ORDER}, 0
    total_sum = int(df["total"].sum())
    if USE_WEIGHTED_MEAN:
        w = df["total"].fillna(0)
        denom = w.sum()
        out = {}
        for k in BIAS_ORDER:
            v = df[k].fillna(0)
            out[k] = float((v * w).sum() / denom) if denom > 0 else float("nan")
        return out, total_sum
    else:
        out = {k: float(df[k].dropna().mean()) if df[k].notna().any() else float("nan") for k in BIAS_ORDER}
        return out, total_sum


def _ensure_full_months(wide: pd.DataFrame, all_months: pd.Index) -> pd.DataFrame:
    """确保每月都有一行；缺失填 0（比例缺失当 0，total 也为 0）"""
    ref = pd.DataFrame({"month": all_months}).merge(wide, on="month", how="left")
    for k in BIAS_ORDER + ["total"]:
        if k in ref.columns:
            ref[k] = ref[k].fillna(0)
    return ref


def build_sheet_table_for_conflict(bbc_df: pd.DataFrame, gdn_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成单个冲突（RU/IP）的总表（一个 DataFrame，用于一个 sheet）：
    行：Period（起止月） + Media（BBC/Guardian）
    列：各模型的 Left/Centre/Right/N
    - 两家媒体共用同一套分段（按二者合并后的最小~最大月份均分）
    """
    # 模型列
    bbc_models = detect_model_label_cols(bbc_df)
    gdn_models = detect_model_label_cols(gdn_df)
    models = [m for m in MODELS_ORDER if (m in bbc_models and m in gdn_models)]
    if not models:
        # 返回空架子
        return pd.DataFrame(columns=["Period", "Media"])

    # 统一的完整月份范围（两媒体合并）
    min_month = min(bbc_df["month"].min(), gdn_df["month"].min())
    max_month = max(bbc_df["month"].max(), gdn_df["month"].max())
    all_months = month_range(min_month, max_month)

    # 均分段并生成段标签（起止月）
    segments = split_months_into_segments(all_months, NUM_SEGMENTS)
    seg_labels = [f"{seg[0]} ~ {seg[-1]}" for seg in segments]

    # 预计算：每媒体、每模型的“月度比例表”（并补齐月份）
    monthly_map = {}  # (media, model) -> wide_df
    for media_name, df_media, model_map in [
        ("BBC", bbc_df, bbc_models),
        ("Guardian", gdn_df, gdn_models),
    ]:
        for model in models:
            wide = monthly_proportions(df_media, model_map[model])
            wide = _ensure_full_months(wide, all_months)
            monthly_map[(media_name, model)] = wide

    # 逐段逐媒体汇总
    rows = []
    for seg_label, seg_months in zip(seg_labels, segments):
        for media_name in ["BBC", "Guardian"]:
            row = {"Period": seg_label, "Media": media_name}
            for model in models:
                wide = monthly_map[(media_name, model)]
                props, nsum = segment_weighted_mean(wide, seg_months)
                for k in BIAS_ORDER:
                    row[f"{model}_{k}"] = props[k]
                row[f"{model}_N"] = nsum
            rows.append(row)

    # 列顺序：Period, Media, 然后每个模型的 Left/Centre/Right/N
    cols = ["Period", "Media"]
    for model in models:
        for k in BIAS_ORDER:
            cols.append(f"{model}_{k}")
        cols.append(f"{model}_N")

    out = pd.DataFrame(rows)
    out = out[cols]
    return out


def main():
    # 载入并生成 month 列
    bbc_ru = load_csv_and_add_month(BBC_RU)
    gdn_ru = load_csv_and_add_month(GUA_RU)
    bbc_ip = load_csv_and_add_month(BBC_IP)
    gdn_ip = load_csv_and_add_month(GUA_IP)

    # 两个冲突 -> 两张表（每张表里含 BBC 与 Guardian）
    sheet_ru = build_sheet_table_for_conflict(bbc_ru, gdn_ru)
    sheet_ip = build_sheet_table_for_conflict(bbc_ip, gdn_ip)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        sheet_ru.to_excel(w, sheet_name="Russia-Ukraine", index=False)
        sheet_ip.to_excel(w, sheet_name="Israel-Hamas", index=False)

    print("已保存：", OUT_XLSX.resolve())


if __name__ == "__main__":
    main()
