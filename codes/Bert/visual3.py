# make_bias_plots_events_top_bottom.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms

# ========= 基本配置 =========
DATA_PATHS = {
    "bbc_ru": "bbc_ru_bias_merge_result_filtered_with_time(1).csv",
    "bbc_ip": "bbc_bias_ip_merge_result_filtered_with_time(1).csv",
    "guardian_ip": "guardian_ip_models_merged_http_filtered_cleaned.csv",
    "guardian_ru": "guardian_ur2_models_merged_http_filtered_cleaned.csv",
}
OUTPUT_DIR = Path("one_axis_events_top_bottom")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

XTICK_MONTH_INTERVAL = 3
YTICKS = np.arange(-1.0, 1.1, 0.25)
ROLL_WINDOW = 3
WEEK_FREQ = "W-MON"

EVENTS = [
    ("2021-08-15", "US withdraw \nfrom Afganistan"),
    ("2022-02-24", "Russia invasion \nof Ukraine"),
    ("2023-10-07", "Israel Hamas \nconflict begins"),
]

# ========= 通用函数 =========
def read_any(path):
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    for enc in ["utf-8", "utf-8-sig", "ISO-8859-1"]:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(p, errors="ignore")

def find_date_column(df):
    for c in df.columns:
        if any(k in c.lower() for k in ["time", "date"]):
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().sum() > 0:
                return c, dt
    for c in df.columns:
        try:
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().sum() > 0:
                return c, dt
        except Exception:
            continue
    raise ValueError("No valid date column")

def extract_label_columns(df):
    model_keys = {"bert": ["bert"], "deepseek": ["deepseek", "deep-seek"], "gemini": ["gemini"]}
    label_cols = {}
    for col in df.columns:
        lc = col.lower()
        if "label" in lc:
            for model, keys in model_keys.items():
                if any(k in lc for k in keys):
                    label_cols[model] = col
    if "bert" not in label_cols:
        for col in df.columns:
            if "predicted_label" in col.lower():
                label_cols["bert"] = col
                break
    return label_cols

def map_labels(series):
    mapping = {"left": -1, "centre": 0, "center": 0, "neutral": 0, "right": 1}
    def norm(x):
        if pd.isna(x): return np.nan
        return mapping.get(re.sub(r"[^a-z]", "", str(x).lower()), np.nan)
    return series.map(norm)

def weekly_mean_index(df):
    col, dt = find_date_column(df)
    df = df.copy()
    df["_dt"] = pd.to_datetime(dt).dt.tz_convert(None)
    label_cols = extract_label_columns(df)
    result = {}
    for model, col in label_cols.items():
        idx = map_labels(df[col])
        tmp = pd.DataFrame({"dt": df["_dt"], "idx": idx}).dropna()
        if tmp.empty: continue
        s = (
            tmp.set_index("dt")
               .resample(WEEK_FREQ)["idx"].mean()
               .rolling(ROLL_WINDOW, center=True).mean()
        )
        result[model] = s
    return result

def combine(series_dict, model, conflict):
    bkey = f"bbc_{conflict}"
    gkey = f"guardian_{conflict}"
    parts = []
    if bkey in series_dict and model in series_dict[bkey]:
        parts.append(series_dict[bkey][model])
    if gkey in series_dict and model in series_dict[gkey]:
        parts.append(series_dict[gkey][model])
    if not parts: return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1)

# ========= 绘图 =========
def plot_model(model, title, series_dict):
    ru = combine(series_dict, model, "ru")
    ip = combine(series_dict, model, "ip")

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.subplots_adjust(bottom=0.22)

    if not ru.empty:
        ax.plot(ru.index, ru.values, label="Ukraine-Russia", color="royalblue", linewidth=1.2)
    if not ip.empty:
        ax.plot(ip.index, ip.values, label="Israel-Hamas", color="orange", linewidth=1.2)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)

    ax.set_title(f"{title} Bias Index Over Time", fontsize=13, pad=8)
    ax.set_ylabel("Bias index (Left=-1, Centre=0, Right=1)", fontsize=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_yticks(YTICKS)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=XTICK_MONTH_INTERVAL))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, fontsize=8)

    # 竖线贯穿全轴
    line_kw = dict(color="gray", linestyle="--", linewidth=1.1, alpha=0.9, zorder=1)
    bbox_kw = dict(boxstyle="round,pad=0.18", fc="white", ec="lightgray", alpha=0.95)
    xy_trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    # 不同模型：事件文字位置
    if model.lower() in ["deepseek", "gemini"]:
        # 放在 0.00 虚线上方一点
        EVENT_Y_FRACTION = 0.56
    else:
        # 放在 x 轴附近
        EVENT_Y_FRACTION = 0.02

    for d, text in EVENTS:
        x = pd.Timestamp(d)
        ax.axvline(x, ymin=0.0, ymax=1.0, **line_kw)
        ax.text(
            x, EVENT_Y_FRACTION, text,
            transform=xy_trans, rotation=90,
            ha="center", va="bottom",
            fontsize=7.8, color="dimgray",
            bbox=bbox_kw, clip_on=False, zorder=3
        )

    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3, linestyle=":")
    fig.savefig(OUTPUT_DIR / f"{model}_bias_index_events_top_bottom.png", bbox_inches="tight")
    plt.close(fig)

# ========= 主流程 =========
dfs = {k: read_any(v) for k, v in DATA_PATHS.items()}
series_dict = {k: weekly_mean_index(df) for k, df in dfs.items()}

plot_model("bert", "BERT Model", series_dict)
plot_model("deepseek", "DeepSeek Model", series_dict)
plot_model("gemini", "Gemini Model", series_dict)

print("✅ Figures saved to:", OUTPUT_DIR.resolve())
