# make_bias_summary_table.py
# 依赖：pip install pandas openpyxl
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ========= 四个文件路径 =========
BBC_RU = "bbc_ru_bias_merge_result_filtered_with_time.csv"
GUA_RU = "guardian_ur2_models_merged_http_filtered_cleaned.csv"
BBC_IP = "bbc_bias_ip_merge_result_filtered_with_time.csv"
GUA_IP = "guardian_ip_models_merged_http_filtered_cleaned.csv"

# ========= 分界日期 =========
WAR_START = {
    "IP": pd.to_datetime("2023-10-07"),  # 以色列–哈马斯
    "RU": pd.to_datetime("2022-02-24"),  # 俄乌战争
}

# ========= 常量 =========
CATEGORIES   = ["Left", "Centre", "Right"]
PERIODS      = ["Overall", "Pre-war", "During-war"]
MODELS_ORDER = ["BERT", "DeepSeek", "Gemini"]   # 忽略 Claude 固定顺序
DATE_CANDS   = ["published_date", "time_std", "time", "date", "datetime", "pub_time", "pub_date"]

# ========= 工具函数 =========
def load_csv_with_date(path: str) -> pd.DataFrame:
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
    return df.dropna(subset=["published_date"])

def normalize_label(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if "left" in s: return "Left"
    if "centre" in s or "center" in s: return "Centre"
    if "right" in s: return "Right"
    return None

def detect_model_label_cols(df: pd.DataFrame) -> Dict[str, str]:
    model_map = {}
    for c in df.columns:
        lc = c.lower()
        if "label" not in lc: continue
        if "url" in lc or "title" in lc: continue
        if "bert" in lc:      model_map["BERT"] = c
        elif "deepseek" in lc:model_map["DeepSeek"] = c
        elif "gemini" in lc:  model_map["Gemini"] = c
        elif "claude" in lc:  model_map["Claude"] = c
    return model_map

def proportions_and_count(series: pd.Series) -> Tuple[Dict[str, float], int]:
    s = series.dropna()
    n = int(s.shape[0])
    props = (s.value_counts(normalize=True) if n > 0 else pd.Series(dtype=float)).to_dict()
    props_full = {c: float(props.get(c, 0.0)) for c in CATEGORIES}
    return props_full, n

def compute_for_one(df: pd.DataFrame, label_col: str, cutoff: pd.Timestamp) -> Dict[str, Dict]:
    s = df[label_col].map(normalize_label)
    overall_props, overall_n = proportions_and_count(s)
    pre_mask     = df["published_date"] < cutoff
    during_mask  = df["published_date"] >= cutoff
    pre_props, pre_n       = proportions_and_count(s[pre_mask])
    during_props, during_n = proportions_and_count(s[during_mask])
    return {
        "Overall":    {"props": overall_props, "n": overall_n},
        "Pre-war":    {"props": pre_props,    "n": pre_n},
        "During-war": {"props": during_props, "n": during_n},
    }

# ========= 核心：一个文件 -> 一个“模型为行”的表 =========
def build_single_file_table(df: pd.DataFrame, war: str, model_map: Dict[str, str]) -> pd.DataFrame:
    """
    输出 DataFrame：
      行：Model（BERT/DeepSeek/Gemini）
      列：Period_Category（如 Overall_Left, Pre-war_Centre, ..., During-war_N）
      值：Left/Centre/Right 为比例(0~1)，N 为样本数
    """
    cutoff = WAR_START[war]
    models = [m for m in MODELS_ORDER if m in model_map]

    records: List[Dict] = []
    for model in models:
        col = model_map[model]
        stats = compute_for_one(df, col, cutoff)
        for period in PERIODS:
            props = stats[period]["props"]
            for cat in CATEGORIES:
                records.append({"Model": model, "Period": period, "Category": cat, "Value": props[cat]})
            records.append({"Model": model, "Period": period, "Category": "N", "Value": stats[period]["n"]})

    tidy = pd.DataFrame(records)
    tidy["colname"] = tidy["Period"] + "_" + tidy["Category"]

    wide = tidy.pivot_table(index="Model", columns="colname", values="Value", aggfunc="first").reset_index()

    # 列顺序：PERIODS × (Left, Centre, Right, N)
    col_order = ["Model"]
    for p in PERIODS:
        for c in (CATEGORIES + ["N"]):
            col_order.append(f"{p}_{c}")
    wide = wide[[c for c in col_order if c in wide.columns]]
    return wide

# ========= 主流程 =========
def main():
    # 读取四文件
    df_bbc_ru = load_csv_with_date(BBC_RU)
    df_gua_ru = load_csv_with_date(GUA_RU)
    df_bbc_ip = load_csv_with_date(BBC_IP)
    df_gua_ip = load_csv_with_date(GUA_IP)

    bbc_models_ru  = detect_model_label_cols(df_bbc_ru)
    gua_models_ru  = detect_model_label_cols(df_gua_ru)
    bbc_models_ip  = detect_model_label_cols(df_bbc_ip)
    gua_models_ip  = detect_model_label_cols(df_gua_ip)

    # 四个表：每个文件一个（忽略 Claude）
    tables = {}
    tables["BBC_RU"]      = build_single_file_table(df_bbc_ru, "RU", bbc_models_ru)
    tables["Guardian_RU"] = build_single_file_table(df_gua_ru, "RU", gua_models_ru)
    tables["BBC_IP"]      = build_single_file_table(df_bbc_ip, "IP", bbc_models_ip)
    tables["Guardian_IP"] = build_single_file_table(df_gua_ip, "IP", gua_models_ip)

    # 合并到一个 Excel（四个 sheet）
    out_xlsx = Path("bias_summary_4tables.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        for name, tdf in tables.items():
            tdf.to_excel(w, sheet_name=name, index=False)
    print("四个表格已保存到：", out_xlsx.resolve())

if __name__ == "__main__":
    main()
