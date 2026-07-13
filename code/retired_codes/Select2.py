# make_bias_summary_by_war.py
# 依赖：pip install pandas openpyxl
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 四个文件路径
BBC_RU = "bbc_ru_bias_merge_result_filtered_with_time.csv"
GUA_RU = "guardian_ur2_models_merged_http_filtered_cleaned.csv"
BBC_IP = "bbc_bias_ip_merge_result_filtered_with_time.csv"
GUA_IP = "guardian_ip_models_merged_http_filtered_cleaned.csv"

# 分界日期
WAR_START = {
    "IP": pd.to_datetime("2023-10-07"),
    "RU": pd.to_datetime("2022-02-24"),
}

# 常量
CATEGORIES   = ["Left", "Centre", "Right"]
PERIODS      = ["Overall", "Pre-war", "During-war"]
MODELS_ORDER = ["BERT", "DeepSeek", "Gemini"]   # 忽略 Claude
DATE_CANDS   = ["published_date", "time_std", "time", "date", "datetime", "pub_time", "pub_date"]

def load_csv_with_date(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string", low_memory=False)
    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    else:
        col_found = None
        lower = {c.lower(): c for c in df.columns}
        for cand in DATE_CANDS:
            if cand in lower:
                col_found = lower[cand]; break
        if col_found is None:
            for c in df.columns:
                lc = c.lower()
                if any(tok in lc for tok in ["date","time","publish","datetime","created","updated"]):
                    col_found = c; break
        if col_found is None:
            raise ValueError(f"{path} 未找到日期列")
        df["published_date"] = pd.to_datetime(df[col_found], errors="coerce")
    return df.dropna(subset=["published_date"])

def normalize_label(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip().lower()
    if "left" in s: return "Left"
    if "centre" in s or "center" in s: return "Centre"
    if "right" in s: return "Right"
    return None

def detect_model_label_cols(df: pd.DataFrame) -> Dict[str,str]:
    m = {}
    for c in df.columns:
        lc = c.lower()
        if "label" not in lc: continue
        if "url" in lc or "title" in lc: continue
        if "bert" in lc: m["BERT"] = c
        elif "deepseek" in lc: m["DeepSeek"] = c
        elif "gemini" in lc: m["Gemini"] = c
    return m

def proportions_and_count(s: pd.Series) -> Tuple[Dict[str,float],int]:
    s = s.dropna()
    n = int(s.shape[0])
    props = (s.value_counts(normalize=True) if n>0 else pd.Series(dtype=float)).to_dict()
    return {c:float(props.get(c,0.0)) for c in CATEGORIES}, n

def compute_for_one(df: pd.DataFrame, label_col: str, cutoff: pd.Timestamp) -> Dict[str,Dict]:
    s = df[label_col].map(normalize_label)
    overall_props, overall_n = proportions_and_count(s)
    pre_props, pre_n = proportions_and_count(s[df["published_date"] < cutoff])
    during_props, during_n = proportions_and_count(s[df["published_date"] >= cutoff])
    return {
        "Overall":    {"props": overall_props, "n": overall_n},
        "Pre-war":    {"props": pre_props,    "n": pre_n},
        "During-war": {"props": during_props, "n": during_n},
    }

def build_table_for_war(bbc_df: pd.DataFrame, gua_df: pd.DataFrame,
                        war: str, bbc_map: Dict[str,str], gua_map: Dict[str,str]) -> pd.DataFrame:
    cutoff = WAR_START[war]
    models = [m for m in MODELS_ORDER if m in bbc_map and m in gua_map]

    records = []
    for model in models:
        bbc_stats = compute_for_one(bbc_df, bbc_map[model], cutoff)
        gua_stats = compute_for_one(gua_df, gua_map[model], cutoff)

        row = {"Model": model}
        for media, stats in [("BBC", bbc_stats), ("Guardian", gua_stats)]:
            for period in PERIODS:
                props = stats[period]["props"]
                for cat in CATEGORIES:
                    row[f"{media}_{period}_{cat}"] = props[cat]
                row[f"{media}_{period}_N"] = stats[period]["n"]
        records.append(row)

    df = pd.DataFrame(records)
    # 列排序
    cols = ["Model"]
    for media in ["BBC","Guardian"]:
        for p in PERIODS:
            for c in CATEGORIES+["N"]:
                cols.append(f"{media}_{p}_{c}")
    return df[[c for c in cols if c in df.columns]]

def main():
    df_bbc_ru = load_csv_with_date(BBC_RU)
    df_gua_ru = load_csv_with_date(GUA_RU)
    df_bbc_ip = load_csv_with_date(BBC_IP)
    df_gua_ip = load_csv_with_date(GUA_IP)

    bbc_models_ru = detect_model_label_cols(df_bbc_ru)
    gua_models_ru = detect_model_label_cols(df_gua_ru)
    bbc_models_ip = detect_model_label_cols(df_bbc_ip)
    gua_models_ip = detect_model_label_cols(df_gua_ip)

    table_ru = build_table_for_war(df_bbc_ru, df_gua_ru, "RU", bbc_models_ru, gua_models_ru)
    table_ip = build_table_for_war(df_bbc_ip, df_gua_ip, "IP", bbc_models_ip, gua_models_ip)

    out_xlsx = Path("bias_summary_by_war.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        table_ru.to_excel(w, sheet_name="Russia-Ukraine", index=False)
        table_ip.to_excel(w, sheet_name="Israel-Hamas", index=False)

    print("已保存：", out_xlsx.resolve())

if __name__ == "__main__":
    main()
