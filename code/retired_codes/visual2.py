import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

# ===============================
# 全局字体与风格（统一放大）
# ===============================
plt.rcParams.update({
    "font.size": 12,          # 基础字体
    "axes.titlesize": 16,     # 子图标题
    "axes.labelsize": 14,     # 坐标轴标题
    "xtick.labelsize": 12,    # x轴刻度
    "ytick.labelsize": 12,    # y轴刻度
    "legend.fontsize": 12,    # 图例文字
})
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.05)  # 可调为 1.15 或 1.2 进一步放大

# ========== 输出目录 ==========
output_dir = "newoutput"
os.makedirs(output_dir, exist_ok=True)

# ========== 读取数据 ==========
df_bbc_ip = pd.read_csv("bbc_bias_ip_merge_result_filtered_with_time.csv")
df_bbc_ru = pd.read_csv("bbc_ru_bias_merge_result_filtered_with_time.csv")
df_guardian_ip = pd.read_excel("guardian_ip_models_merged_http_filtered_cleaned.xlsx")
df_guardian_ru = pd.read_excel("guardian_ur2_models_merged_http_filtered_cleaned.xlsx")

# ========== 标签标准化 ==========
for df in [df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip]:
    for col in df.columns:
        if col.startswith('predicted_label'):
            df[col] = df[col].astype(str).str.title().str.strip()

# ========== 添加 Media 和 War ==========
df_bbc_ru["Media"] = "BBC"; df_bbc_ru["War"] = "Russia-Ukraine"
df_bbc_ip["Media"] = "BBC"; df_bbc_ip["War"] = "Israel-Palestine"
df_guardian_ru["Media"] = "Guardian"; df_guardian_ru["War"] = "Russia-Ukraine"
df_guardian_ip["Media"] = "Guardian"; df_guardian_ip["War"] = "Israel-Palestine"

# ========== 划分战前/战后（Guardian 用发布时间；BBC 用启发式） ==========
df_guardian_ru["published_date"] = pd.to_datetime(df_guardian_ru["published_date"])
df_guardian_ip["published_date"] = pd.to_datetime(df_guardian_ip["published_date"])
df_guardian_ru["Period"] = df_guardian_ru["published_date"].apply(
    lambda d: "Pre-War" if d < pd.Timestamp("2022-02-24") else "Post-War"
)
df_guardian_ip["Period"] = df_guardian_ip["published_date"].apply(
    lambda d: "Pre-War" if d < pd.Timestamp("2023-10-07") else "Post-War"
)

def classify_bbc_period(url, title, war_start_id):
    match = re.search(r"(\d+)$", str(url))
    id_val = int(match.group(1)) if match else None
    title_low = str(title).lower()
    if id_val:
        return "Pre-War" if id_val < war_start_id else "Post-War"
    if "if russia invades" in title_low:
        return "Pre-War"
    if any(w in title_low for w in [" war", "invasion", "attack", "ceasefire"]):
        return "Post-War"
    return "Pre-War"

df_bbc_ru["Period"] = df_bbc_ru.apply(lambda row: classify_bbc_period(row["url"], row["title"], 60400000), axis=1)
df_bbc_ip["Period"] = df_bbc_ip.apply(lambda row: classify_bbc_period(row["url"], row["title"], 67000000), axis=1)

# ========== 偏见得分映射（移除 Claude） ==========
bias_map = {"Left": -1, "Center": 0, "Right": 1}
selected_models = ["bert", "deepseek", "gemini"]  # 仅保留三种模型

for df in [df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip]:
    for model in selected_models:
        col = f"predicted_label_{model}"
        if col in df.columns:
            df[f"score_{model}"] = df[col].map(bias_map)

# ========== 合并 + 平均得分 + 模型名映射 ==========
df_all = pd.concat([df_bbc_ru, df_bbc_ip, df_guardian_ru, df_guardian_ip], ignore_index=True)
score_cols = [f"score_{m}" for m in selected_models]

df_long = df_all.melt(
    id_vars=["Media", "War", "Period"],
    value_vars=score_cols,
    var_name="Model", value_name="Score"
)
# 原始列名如 score_bert -> bert -> Title() -> Bert
df_long["Model"] = df_long["Model"].str.replace("score_", "").str.title()
df_long = df_long.dropna(subset=["Score"])

# ✅ 模型名称显示映射（用于图中标签/图例）
model_name_map = {"Bert": "BERT", "Deepseek": "DeepSeek", "Gemini": "Gemini"}
df_long["Model"] = df_long["Model"].map(model_name_map).fillna(df_long["Model"])

avg_scores = df_long.groupby(["Media", "War", "Period", "Model"])["Score"].mean().reset_index()
avg_scores["War"] = avg_scores["War"].replace({"Israel-Palestine": "Hamas-Israel"})
avg_scores["Scenario"] = avg_scores["War"] + "\n" + avg_scores["Period"]
scenario_order = [
    "Russia-Ukraine\nPre-War", "Russia-Ukraine\nPost-War",
    "Hamas-Israel\nPre-War", "Hamas-Israel\nPost-War"
]
avg_scores["Scenario"] = pd.Categorical(avg_scores["Scenario"], categories=scenario_order, ordered=True)

# 统一的模型顺序与调色板（与映射后的名称一致）
MODEL_ORDER = ["BERT", "DeepSeek", "Gemini"]
PALETTE = {"BERT": "#1f77b4", "DeepSeek": "#2ca02c", "Gemini": "#d62728"}

# ========== 保存平均表格 ==========
avg_scores.to_csv(os.path.join(output_dir, "average_bias_score_table.csv"), index=False)

# ========== 柱状图 ==========
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=avg_scores,
    x="Scenario", y="Score", hue="Model",
    hue_order=MODEL_ORDER,
    palette=PALETTE
)
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("War & Period", fontsize=14)
plt.ylabel("Average Bias Score\n(-1 = Left, 0 = Center, 1 = Right)", fontsize=14)
plt.title("Average Bias Score by Model (BBC vs Guardian, Pre- vs Post-War)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ✅ 图例放在右上角、图外侧
leg = plt.legend(
    title="Model",
    fontsize=12,
    title_fontsize=13,
    loc="upper left",           # 锚点相对位置
    bbox_to_anchor=(1.02, 1),   # 把图例移到绘图区右上角之外
    frameon=True,
    borderpad=0.6,
    labelspacing=0.4,
    handlelength=1.8
)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "average_bias_score_barplot_no_labels.png"),
    dpi=300,
    bbox_inches="tight"          # ✅ 保证保存时把图外图例也包含进去
)
plt.close()


# ========== 折线图 ==========
def plot_model_trends(avg_scores: pd.DataFrame, output_path: str):
    models = MODEL_ORDER
    colors = PALETTE
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    time_points = [0, 1]

    for i, war in enumerate(["Russia-Ukraine", "Hamas-Israel"]):
        for j, media in enumerate(["BBC", "Guardian"]):
            ax = axes[i, j]
            data = avg_scores[(avg_scores["War"] == war) & (avg_scores["Media"] == media)]
            for model in models:
                pre_val = data[(data["Model"] == model) & (data["Period"] == "Pre-War")]["Score"]
                post_val = data[(data["Model"] == model) & (data["Period"] == "Post-War")]["Score"]
                if len(pre_val) == 0 or len(post_val) == 0:
                    continue
                y_vals = [pre_val.iloc[0], post_val.iloc[0]]
                ax.plot(time_points, y_vals, marker="o", linewidth=2, color=colors[model], label=model)

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xticks(time_points)
            ax.set_xticklabels(["Pre-War", "Post-War"], fontsize=12)
            ax.set_ylim(-0.6, 0.6)
            if j == 0:
                ax.set_ylabel("Bias Score", fontsize=14)
            ax.set_title(f"{media} - {war} War", fontsize=15)
            ax.tick_params(axis='y', labelsize=12)

    # 统一图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Model", loc="upper center", ncol=3,
                   fontsize=12, title_fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=300)
    plt.close()

plot_model_trends(avg_scores, os.path.join(output_dir, "bias_score_trend_lineplot.png"))

# ========== 热力图 + 输出表格 ==========
def plot_heatmaps_and_export(avg_scores: pd.DataFrame, output_dir: str):
    bbc_data = avg_scores[avg_scores['Media'] == 'BBC'].pivot(index='Model', columns=['War', 'Period'], values='Score')
    guardian_data = avg_scores[avg_scores['Media'] == 'Guardian'].pivot(index='Model', columns=['War', 'Period'], values='Score')
    new_cols = [('Russia-Ukraine','Pre-War'),('Russia-Ukraine','Post-War'),
                ('Hamas-Israel','Pre-War'),('Hamas-Israel','Post-War')]
    bbc_data = bbc_data.reindex(columns=new_cols)
    guardian_data = guardian_data.reindex(columns=new_cols)
    bbc_data.columns = ['RU Pre', 'RU Post', 'HI Pre', 'HI Post']
    guardian_data.columns = ['RU Pre', 'RU Post', 'HI Pre', 'HI Post']

    # 行顺序按映射后的名称排序，确保一致性（可选）
    bbc_data = bbc_data.reindex(MODEL_ORDER)
    guardian_data = guardian_data.reindex(MODEL_ORDER)

    bbc_data.to_csv(os.path.join(output_dir, "bbc_bias_score_matrix.csv"))
    guardian_data.to_csv(os.path.join(output_dir, "guardian_bias_score_matrix.csv"))

    plt.figure(figsize=(11.5, 4.8))

    # BBC
    plt.subplot(1, 2, 1)
    ax1 = sns.heatmap(
        bbc_data, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-0.6, vmax=0.6, cbar=False,
        annot_kws={"size": 12}
    )
    plt.title("BBC", fontsize=16)
    plt.ylabel("Model", fontsize=14)
    plt.xlabel("War and Period", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Guardian
    plt.subplot(1, 2, 2)
    ax2 = sns.heatmap(
        guardian_data, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-0.6, vmax=0.6,
        cbar=True, cbar_kws={"label": "Bias Score"},
        annot_kws={"size": 12}
    )
    plt.title("Guardian", fontsize=16)
    plt.xlabel("War and Period", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # 调整注释文字颜色对比度
    for ax in [ax1, ax2]:
        for text in ax.texts:
            try:
                val = float(text.get_text())
                text.set_color("white" if abs(val) >= 0.4 else "black")
            except:
                pass

    # 调大色条刻度与标签
    cbar = ax2.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Bias Score", fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_score_heatmaps.png"), dpi=300)
    plt.close()

plot_heatmaps_and_export(avg_scores, output_dir)

# ========== 雷达图 ==========
def plot_model_radar_charts_and_export(df_long: pd.DataFrame, output_dir: str):
    # 计算雷达图数据并导出
    avg_by_war = df_long.groupby(['Media', 'War', 'Model'])['Score'].mean().reset_index()
    radar_table = avg_by_war.pivot(index='Model', columns=['Media', 'War'], values='Score')
    # 行顺序固定
    radar_table = radar_table.reindex(MODEL_ORDER)
    radar_table.to_csv(os.path.join(output_dir, "model_media_war_radar_data.csv"))

    # 轴顺序：四个点
    axis_order = [('BBC', 'Russia-Ukraine'), ('Guardian', 'Russia-Ukraine'),
                  ('Guardian', 'Hamas-Israel'), ('BBC', 'Hamas-Israel')]
    models = MODEL_ORDER
    model_colors = PALETTE

    fig, axes = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(12.5, 4.6))
    angles = np.linspace(0, 2 * np.pi, len(axis_order) + 1)

    for ax, model in zip(axes, models):
        data = []
        for (media, war) in axis_order:
            score = avg_by_war[(avg_by_war['Media'] == media) &
                               (avg_by_war['War'] == war) &
                               (avg_by_war['Model'] == model)]['Score'].values
            val = score[0] if len(score) > 0 else 0.0
            data.append(val + 1)  # 偏移到 [0,2] 便于雷达展示
        data.append(data[0])     # 闭合

        ax.plot(angles, data, linewidth=2.5, color=model_colors[model])
        ax.fill(angles, data, alpha=0.20, color=model_colors[model])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['BBC RU', 'Guardian RU', 'Guardian HI', 'BBC HI'], fontsize=12)
        ax.set_ylim(0, 2)
        ax.set_yticks([0.5, 1.0, 1.5, 2.0])
        ax.set_yticklabels(['-0.5', '0', '0.5', '1.0'], fontsize=11)
        ax.set_title(model, y=1.10, fontsize=15)
        ax.set_rlabel_position(225)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_bias_radar_charts.png"), dpi=300)
    plt.close()

plot_model_radar_charts_and_export(df_long, output_dir)

print("All figures and tables saved to:", output_dir)
