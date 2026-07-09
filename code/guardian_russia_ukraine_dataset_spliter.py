import pandas as pd

# read CSV file
df = pd.read_csv("guardian_russia_ukraine_articles.csv")

# convert published_at to datetime 
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

# define cutoff
cutoff_date = pd.to_datetime("2022-02-24").tz_localize("UTC")

# split dataset
df_before = df[df['published_at'] < cutoff_date]
df_after = df[df['published_at'] >= cutoff_date]

# save to sub sets
df_before.to_csv("guardian_russia_ukraine_before_war.csv", index=False, encoding='utf-8-sig')
df_after.to_csv("guardian_russia_ukraine_during_war.csv", index=False, encoding='utf-8-sig')

# Total rows
print(f"✅ DONE：")
print(f"  🔹 prior to 2022-02-24 data: {len(df_before)}")
print(f"  🔸 2022-02-24 onwards data: {len(df_after)}")
