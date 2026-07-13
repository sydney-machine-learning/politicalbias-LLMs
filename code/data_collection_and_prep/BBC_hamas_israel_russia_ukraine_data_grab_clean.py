from datasets import load_dataset
import pandas as pd

# ========== 时间区间 ==========
years = list(range(2020, 2025))
months = [f"{y}-{m:02d}" for y in years for m in range(1, 13)]

# ========== 关键词 ==========
keywords_ukraine = ['ukraine', 'russia', 'putin', 'zelensky', 'donbas', 'kyiv', 'moscow']
keywords_israel = ['israel', 'palestine', 'gaza', 'hamas', 'idf', 'west bank', 'netanyahu']

# ========== 临时存放每月筛选结果 ==========
ukraine_articles = []
israel_articles = []

for month in months:
    try:
        print(f"📥 Loading: {month}")
        ds = load_dataset("RealTimeData/bbc_news_alltime", month, split="train")
        df = ds.to_pandas()

        # 确保包含必要字段
        if not {'description', 'content'}.issubset(df.columns):
            print(f"⚠️  Skipping {month} (missing description/content)")
            continue

        # 合并字段用于关键词匹配
        combined_text = df['description'].fillna('') + ' ' + df['content'].fillna('')

        # 俄乌关键词匹配
        ukraine_mask = combined_text.str.contains('|'.join(keywords_ukraine), case=False, na=False)
        df_ukraine = df[ukraine_mask]

        # 巴以关键词匹配
        israel_mask = combined_text.str.contains('|'.join(keywords_israel), case=False, na=False)
        df_israel = df[israel_mask]

        # 保存本月筛选结果
        if not df_ukraine.empty:
            ukraine_articles.append(df_ukraine)
        if not df_israel.empty:
            israel_articles.append(df_israel)

        print(f"  ✅ Ukraine: {len(df_ukraine)}, Israel/Palestine: {len(df_israel)}")

    except Exception as e:
        print(f"❌ Failed to load {month}: {e}")

# ========== 合并并写出 CSV ==========
print("\n🔗 Merging results...")
df_ukraine_all = pd.concat(ukraine_articles, ignore_index=True)
df_israel_all = pd.concat(israel_articles, ignore_index=True)

df_ukraine_all.to_csv("bbc_ukraine_2020_2024.csv", index=False)
df_israel_all.to_csv("bbc_israel_2020_2024.csv", index=False)

print("✅ DONE:")
print("  → bbc_ukraine_2020_2024.csv")
print("  → bbc_israel_2020_2024.csv")