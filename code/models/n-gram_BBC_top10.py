import os
import re
import time
import pickle
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 画图样式：无网格、大字号 =====
SCALE = 1.5  # ← 想再调大/调小，改这里

TITLE_SIZE = int(18 * SCALE)
TICK_SIZE_X = int(14 * SCALE)
TICK_SIZE_Y = int(18 * SCALE)
LABEL_SIZE  = int(14 * SCALE)
ANNOT_SIZE  = int(12 * SCALE)

sns.set(style="white")
plt.rcParams.update({
    "axes.labelcolor": "black",
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "normal",  # 不加粗
    "xtick.labelsize": TICK_SIZE_X,
    "ytick.labelsize": TICK_SIZE_Y,
})

# ===== spaCy 英文模型 =====
nlp = spacy.load("en_core_web_sm")

# ===== 自定义停用词 =====
custom_stopwords = {
    "say", "gmt", "bst", "report", "update", "day", "state", "year", "old",
    "new", "latest", "live", "breaking", "include", "tell","video","play","javascript","browser","enable","need","watch",
    "bbc","news","subscribe","cookie","cookies","app"
}

# ===== BBC 数据集（红框这两份）=====
files = {
    "Israel-Hamas": "bbc_israel_2020_2024_advanced_filtered_filtered_upto_10000_tokens_dedup.csv",
    "Russia-Ukraine": "bbc_ukraine_2020_2024_advanced_filtered_filtered_upto_10000_tokens_dedup.csv",
}

def sanitize(name: str) -> str:
    """将标签转为安全文件名（下划线、小写）。"""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()

# ===== 文本清洗 + 缓存（pkl 文件带 bbc_ 标注）=====
def spacy_clean_texts(texts, cache_tag: str):
    cache_path = f"bbc_{cache_tag}_cleaned.pkl"   # <-- 关键：bbc_ 前缀
    if os.path.exists(cache_path):
        print(f"📦 Loading cached texts from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"🧹 Cleaning {len(texts)} texts with spaCy...")
    start = time.time()
    cleaned = []
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        tokens = [
            token.lemma_.lower() for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
            and token.lemma_.lower() not in custom_stopwords
        ]
        cleaned.append(" ".join(tokens))
    print(f"✅ Done in {time.time() - start:.2f}s")

    with open(cache_path, "wb") as f:
        pickle.dump(cleaned, f)
    return cleaned

# ===== n-gram 提取（Tri-gram Top10）=====
def generate_trigrams(texts, top_k=10):
    vectorizer = CountVectorizer(ngram_range=(3, 3))
    X = vectorizer.fit_transform(texts)
    sums = X.sum(axis=0)
    freqs = [(word, int(sums[0, idx])) for word, idx in vectorizer.vocabulary_.items()]
    freqs.sort(key=lambda x: x[1], reverse=True)

    def ok(ng):
        return all(w not in custom_stopwords for w in ng.split())

    return [item for item in freqs if ok(item[0])][:top_k]

# ===== 画图（避免右侧数值被裁）=====
def plot_trigrams(trigrams, title, ylabel, save_path=None):
    df_plot = pd.DataFrame(trigrams, columns=[ylabel, "Count"]).sort_values("Count", ascending=True)

    plt.figure(figsize=(12, 8))  # 需要更大画布可改为 (14, 10) 或乘以 SCALE
    norm = plt.Normalize(df_plot["Count"].min(), df_plot["Count"].max())
    colors = plt.cm.Blues(norm(df_plot["Count"]))

    ax = plt.gca()
    bars = ax.barh(df_plot[ylabel], df_plot["Count"], color=colors)

    max_count = float(df_plot["Count"].max())
    ax.set_xlim(0, max_count * 1.12)
    ax.margins(x=0.02)

    # 同步放大刻度
    ax.tick_params(axis='x', labelsize=TICK_SIZE_X)
    ax.tick_params(axis='y', labelsize=TICK_SIZE_Y)

    # 数值标注（也放大 2 倍）
    for bar, val in zip(bars, df_plot["Count"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        if x >= max_count * 0.88:
            ax.text(x - 8, y, f"{int(val)}", va='center', ha='right',
                    fontsize=ANNOT_SIZE, color='white')   # 内置白字
        else:
            ax.text(x + 6, y, f"{int(val)}", va='center', ha='left',
                    fontsize=ANNOT_SIZE, color='black')   # 外置黑字

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='normal', color='black')
    ax.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

    ax.grid(False); plt.grid(False)

    # 左边距适当加大，避免放大后的 y 轴标签被截
    plt.subplots_adjust(left=0.40, right=0.90)  # 原来是 0.32，可按需再调大

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📁 Saved plot to: {save_path}")
    plt.show()
# ===== 主流程：BBC 两个数据集 =====
for label, csv_path in files.items():
    print(f"\n📊 BBC Analyzing {label}")
    df = pd.read_csv(csv_path)
    texts = df["content"].dropna().astype(str).tolist()

    cache_tag = sanitize(label)               # e.g., "israel_hamas"
    cleaned = spacy_clean_texts(texts, cache_tag)  # 生成 bbc_xxx_cleaned.pkl

    trigrams = generate_trigrams(cleaned, top_k=10)
    title = f"BBC - {label}"
    out_png = f"Tri-gram-BBC-{label.replace(' ', '-').replace('/', '-')}-Top10.png"
    plot_trigrams(trigrams, title, ylabel="3-gram", save_path=out_png)