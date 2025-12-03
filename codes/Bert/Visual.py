import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# ====== 文件路径 ======
file_paths = {
    'bbc_ru': 'bbc_ru_bias_merge_result_filtered_with_time.csv',
    'bbc_ip': 'bbc_bias_ip_merge_result_filtered_with_time.csv',
    'guardian_ru': 'guardian_ur2_models_merged_http_filtered_cleaned.xlsx',
    'guardian_ip': 'guardian_ip_models_merged_http_filtered_cleaned.xlsx'
}

# ====== 加载数据 ======
df_bbc_ru = pd.read_csv(file_paths['bbc_ru'])
df_bbc_ip = pd.read_csv(file_paths['bbc_ip'])
df_guardian_ru = pd.read_excel(file_paths['guardian_ru'])
df_guardian_ip = pd.read_excel(file_paths['guardian_ip'])

# ====== 标准化标签格式 ======
for df in [df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip]:
    for col in df.columns:
        if col.startswith('predicted_label'):
            df[col] = df[col].str.title().str.strip()

# ====== 添加媒体与战争类型 ======
df_bbc_ru['Media'] = 'BBC'; df_bbc_ru['War'] = 'Russia-Ukraine'
df_bbc_ip['Media'] = 'BBC'; df_bbc_ip['War'] = 'Israel-Palestine'
df_guardian_ru['Media'] = 'Guardian'; df_guardian_ru['War'] = 'Russia-Ukraine'
df_guardian_ip['Media'] = 'Guardian'; df_guardian_ip['War'] = 'Israel-Palestine'

# ====== Guardian 用发布日期划分战前战后 ======
df_guardian_ru['published_date'] = pd.to_datetime(df_guardian_ru['published_date'], errors='coerce')
df_guardian_ip['published_date'] = pd.to_datetime(df_guardian_ip['published_date'], errors='coerce')
df_guardian_ru['Period'] = df_guardian_ru['published_date'].apply(lambda d: 'Pre-War' if d < pd.Timestamp("2022-02-24") else 'Post-War')
df_guardian_ip['Period'] = df_guardian_ip['published_date'].apply(lambda d: 'Pre-War' if d < pd.Timestamp("2023-10-07") else 'Post-War')

# ====== BBC 根据URL或标题关键词估算战前战后 ======
def classify_bbc_period(url, title, war_start_id):
    match = re.search(r'(\d+)$', str(url))
    try:
        id_val = int(match.group(1)) if match else None
    except ValueError:
        id_val = None
    title_low = str(title).lower()
    if id_val:
        return 'Pre-War' if id_val < war_start_id else 'Post-War'
    if 'if russia invades' in title_low:
        return 'Pre-War'
    if any(w in title_low for w in [' war', 'invasion', 'attack', 'ceasefire']):
        return 'Post-War'
    return 'Pre-War'

df_bbc_ru['Period'] = df_bbc_ru.apply(lambda row: classify_bbc_period(row['url'], row['title'], 60400000), axis=1)
df_bbc_ip['Period'] = df_bbc_ip.apply(lambda row: classify_bbc_period(row['url'], row['title'], 67000000), axis=1)

# ====== 映射标签为偏见分数 ======
bias_map = {'Left': -1, 'Center': 0, 'Right': 1}
for df in [df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip]:
    for model in ['bert', 'claude', 'deepseek', 'gemini']:
        col = f'predicted_label_{model}'
        if col in df.columns:
            df[f'score_{model}'] = df[col].map(bias_map)

# ====== 合并并转换为长格式 ======
df_all = pd.concat([df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip], ignore_index=True)
score_cols = [col for col in df_all.columns if col.startswith('score_')]
df_long = df_all.melt(id_vars=['Media', 'War', 'Period'], value_vars=score_cols,
                      var_name='Model', value_name='Score')
df_long['Model'] = df_long['Model'].str.replace('score_', '').str.title()
df_long = df_long.dropna(subset=['Score'])

# ====== 去除 Claude ======
df_long = df_long[df_long['Model'] != 'Claude']

# ====== 平均得分计算 ======
avg_scores = df_long.groupby(['Media', 'War', 'Period', 'Model'])['Score'].mean().reset_index()
avg_scores['Scenario'] = avg_scores['War'] + '\n' + avg_scores['Period']
scenario_order = ["Russia-Ukraine\nPre-War", "Russia-Ukraine\nPost-War",
                  "Israel-Palestine\nPre-War", "Israel-Palestine\nPost-War"]
avg_scores['Scenario'] = pd.Categorical(avg_scores['Scenario'], categories=scenario_order, ordered=True)

# ====== 绘图 ======
sns.set_style('whitegrid')
palette = {'Bert': '#1f77b4', 'Deepseek': '#2ca02c', 'Gemini': '#d62728'}

plt.figure(figsize=(10, 6))
sns.barplot(x='Scenario', y='Score', hue='Model', data=avg_scores,
            hue_order=['Bert', 'Deepseek', 'Gemini'], palette=palette)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("War & Period")
plt.ylabel("Average Bias Score\n(-1 = Left, 0 = Center, 1 = Right)")
plt.title("Average Bias Score by Model (Claude Excluded)")
plt.legend(title='Model')
plt.tight_layout()

# ====== 保存至 neoutput 文件夹 ======
output_dir = "newoutput"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "avg_bias_score_no_claude.png")
plt.savefig(output_file, dpi=300)
print(f"✅ 图像已保存至: {output_file}")

plt.show()
