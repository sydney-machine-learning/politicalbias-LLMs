import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time

# ===== 图表设置：无网格、放大标签 =====
SCALE = 1.5

TITLE_SIZE = int(18 * SCALE)
TICK_SIZE_X = int(14 * SCALE)
TICK_SIZE_Y = int(18 * SCALE)
LABEL_SIZE  = int(14 * SCALE)
ANNOT_SIZE  = int(12 * SCALE)

sns.set(style="white")
plt.rcParams.update({
    "axes.labelcolor": "black",
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "normal",   # 不加粗
    "xtick.labelsize": TICK_SIZE_X,
    "ytick.labelsize": TICK_SIZE_Y,
})

# 加载 spaCy 英文模型
nlp = spacy.load("en_core_web_sm")

# 自定义停用词（扩展）
custom_stopwords = {
    "say", "gmt", "bst", "report", "update", "day", "state", "year", "old",
    "new", "latest", "live", "breaking", "include", "tell"
}

# 要分析的文件
files = {
    "Israel-Hamas": "guardian_israel_hamas_total_filtered_upto_10000_tokens_cleaned.csv",
    "Russia-Ukraine": "guardian_russia_ukraine_total_filtered_upto_10000_tokens_cleaned.csv"
}

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()

# 文本清洗 + 缓存
def spacy_clean_texts(texts, cache_path):
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

# N-gram 提取（Top10）
def generate_ngrams(texts, n=1, top_k=10):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(texts)
    sums = X.sum(axis=0)
    freqs = [(word, int(sums[0, idx])) for word, idx in vectorizer.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)

    def is_valid_ngram(ng):
        return all(w not in custom_stopwords for w in ng.split())

    return [item for item in freqs if is_valid_ngram(item[0])][:top_k]

# 可视化（无 grid，放大标签）
def plot_trigrams(trigrams, title, ylabel, save_path=None):
    dfp = pd.DataFrame(trigrams, columns=[ylabel, "Count"]).sort_values("Count", ascending=True)

    plt.figure(figsize=(12, 8))
    norm = plt.Normalize(dfp["Count"].min(), dfp["Count"].max())
    colors = plt.cm.Blues(norm(dfp["Count"]))

    ax = plt.gca()
    bars = ax.barh(dfp[ylabel], dfp["Count"], color=colors)

    max_c = float(dfp["Count"].max())
    ax.set_xlim(0, max_c * 1.12)
    ax.margins(x=0.02)

    ax.tick_params(axis='x', labelsize=TICK_SIZE_X)
    ax.tick_params(axis='y', labelsize=TICK_SIZE_Y)

    # 标注：长条内置白字，短条外置黑字
    for bar, val in zip(bars, dfp["Count"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        if x >= max_c * 0.88:
            ax.text(x - 8, y, f"{int(val)}", va='center', ha='right',
                    fontsize=ANNOT_SIZE, color='white')
        else:
            ax.text(x + 6, y, f"{int(val)}", va='center', ha='left',
                    fontsize=ANNOT_SIZE, color='black')

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='normal', color='black')
    ax.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

    # 去掉 grid
    ax.grid(False); plt.grid(False)

    # 放大后给左、右各留空间
    plt.subplots_adjust(left=0.40, right=0.90)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📁 Saved plot to: {save_path}")
    plt.show()

# 主流程（仍然遍历 1/2/3-gram；如只需 tri-gram，把 for n in [3] 即可）
for label, file_path in files.items():
    print(f"\n📊 Analyzing {label}")
    df = pd.read_csv(file_path)
    texts = df["content"].dropna().astype(str).tolist()

    cache_file = f"{label.lower().replace(' ', '_')}_cleaned.pkl"
    cleaned_texts = spacy_clean_texts(texts, cache_file)

    # 只画 Tri-gram（3-gram），Top 10，标题按要求改名
    n = 3
    title = f"The Guardian - {label}"
    col_name = "3-gram"
    top_ngrams = generate_ngrams(cleaned_texts, n=n, top_k=10)

    filename = f"{label.lower().replace(' ', '_')}_trigram_top10.png"
    plot_trigrams(top_ngrams, title, col_name, save_path=filename)