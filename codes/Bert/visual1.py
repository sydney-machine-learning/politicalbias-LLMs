import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# ====== 路径设置 ======
output_dir = "newoutput"
os.makedirs(output_dir, exist_ok=True)

# ====== 读取四个文件 ======
df_bbc_ip = pd.read_csv("bbc_bias_ip_merge_result_filtered_with_time.csv")
df_bbc_ru = pd.read_csv("bbc_ru_bias_merge_result_filtered_with_time.csv")
df_guardian_ip = pd.read_excel("guardian_ip_models_merged_http_filtered_cleaned.xlsx")
df_guardian_ru = pd.read_excel("guardian_ur2_models_merged_http_filtered_cleaned.xlsx")

# ====== 标准化标签 ======
for df in [df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip]:
    for col in df.columns:
        if col.startswith('predicted_label'):
            df[col] = df[col].astype(str).str.title().str.strip()

# ====== 添加 Media 和 War 标志 ======
df_bbc_ru['Media'] = 'BBC';       df_bbc_ru['War'] = 'Russia-Ukraine'
df_bbc_ip['Media'] = 'BBC';       df_bbc_ip['War'] = 'Israel-Palestine'
df_guardian_ru['Media'] = 'Guardian'; df_guardian_ru['War'] = 'Russia-Ukraine'
df_guardian_ip['Media'] = 'Guardian'; df_guardian_ip['War'] = 'Israel-Palestine'

# ====== 添加 Period 划分 ======
# Guardian 使用发布时间
df_guardian_ru['published_date'] = pd.to_datetime(df_guardian_ru['published_date'])
df_guardian_ip['published_date'] = pd.to_datetime(df_guardian_ip['published_date'])
df_guardian_ru['Period'] = df_guardian_ru['published_date'].apply(
    lambda d: 'Pre-War' if d < pd.Timestamp("2022-02-24") else 'Post-War')
df_guardian_ip['Period'] = df_guardian_ip['published_date'].apply(
    lambda d: 'Pre-War' if d < pd.Timestamp("2023-10-07") else 'Post-War')

# BBC 用 URL 中的数字 ID 推断时期
def classify_bbc_period(url, title, war_start_id):
    match = re.search(r'(\d+)$', str(url))
    id_val = int(match.group(1)) if match else None
    title_low = str(title).lower()
    if id_val:
        return 'Pre-War' if id_val < war_start_id else 'Post-War'
    if 'if russia invades' in title_low:
        return 'Pre-War'
    if any(w in title_low for w in [' war', 'invasion', 'attack', 'ceasefire']):
        return 'Post-War'
    return 'Pre-War'

df_bbc_ru['Period'] = df_bbc_ru.apply(
    lambda row: classify_bbc_period(row['url'], row['title'], war_start_id=60400000), axis=1)
df_bbc_ip['Period'] = df_bbc_ip.apply(
    lambda row: classify_bbc_period(row['url'], row['title'], war_start_id=67000000), axis=1)

# ====== 映射标签为得分 ======
bias_map = {'Left': -1, 'Center': 0, 'Right': 1}
for df in [df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip]:
    for model in ['bert', 'claude', 'deepseek', 'gemini']:
        col = f'predicted_label_{model}'
        df[f'score_{model}'] = df[col].map(bias_map)

# ====== 合并数据并转换长格式 ======
df_all = pd.concat([df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip], ignore_index=True)
score_cols = ['score_bert','score_claude','score_deepseek','score_gemini']
existing_score_cols = [col for col in score_cols if col in df_all.columns]

df_long = df_all.melt(
    id_vars=['Media','War','Period'],
    value_vars=existing_score_cols,
    var_name='Model', value_name='Score'
)
df_long['Model'] = df_long['Model'].str.replace('score_', '').str.title()
df_long = df_long.dropna(subset=['Score'])

# ====== 去除 Claude ======
df_long = df_long[df_long['Model'] != 'Claude']

# ====== 平均得分表格 ======
avg_scores = df_long.groupby(['Media','War','Period','Model'])['Score'].mean().reset_index()

# 替换 War 名称为 Hamas-Israel
avg_scores['War'] = avg_scores['War'].replace({'Israel-Palestine': 'Hamas-Israel'})

# 添加 Scenario 列
avg_scores['Scenario'] = avg_scores['War'] + '\n' + avg_scores['Period']
scenario_order = ["Russia-Ukraine\nPre-War", "Russia-Ukraine\nPost-War",
                  "Hamas-Israel\nPre-War", "Hamas-Israel\nPost-War"]
avg_scores['Scenario'] = pd.Categorical(avg_scores['Scenario'], categories=scenario_order, ordered=True)

# 保存表格（去掉Claude的版本）
avg_scores.to_csv(os.path.join(output_dir, "average_bias_score_table_no_claude.csv"), index=False)

# ====== 绘图并保存（无 Claude）======
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
palette = {'Bert':'#1f77b4', 'Deepseek':'#2ca02c', 'Gemini':'#d62728'}

barplot = sns.barplot(
    data=avg_scores,
    x='Scenario', y='Score', hue='Model',
    hue_order=['Bert','Deepseek','Gemini'], palette=palette
)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("War & Period")
plt.ylabel("Average Bias Score\n(-1 = Left, 0 = Center, 1 = Right)")
plt.title("Average Bias Score by Model (Claude Removed)")
plt.legend(title="Model")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_bias_score_barplot_no_claude.png"), dpi=300)
plt.close()
