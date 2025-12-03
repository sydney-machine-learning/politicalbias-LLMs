import pandas as pd

# === 参数配置 ===
input_file = "merged_bias_labels.csv"
output_utf8_file = "merged_bias_labels_cleaned.csv"
output_excel_file = "merged_bias_labels_cleaned_excel.csv"

# === 读取 CSV 文件 ===
df = pd.read_csv(input_file, encoding="ISO-8859-1")  # 防乱码

# === 第一步：去除 url 或 title 缺失的行 ===
df = df.dropna(subset=["url", "title"])

# === 第二步：去除完全重复的行 ===
df = df.drop_duplicates()

# === 第三步：按 url 去重，只保留一条记录 ===
df = df.drop_duplicates(subset=["url"])

# === 第四步：去除所有 predicted_label_x 全为空的行 ===
predicted_cols = [col for col in df.columns if col.startswith("predicted_label_")]
df = df.dropna(subset=predicted_cols, how='all')

# === 第五步：保存为两个版本 ===
# 1. 标准 UTF-8 编码版本
df.to_csv(output_utf8_file, index=False, encoding='utf-8')

# 2. Excel 兼容 UTF-8 with BOM 编码版本
df.to_csv(output_excel_file, index=False, encoding='utf-8-sig')

# === 输出提示信息 ===
print("✅ 清洗完成！")
print(f"- UTF-8 文件: {output_utf8_file}")
print(f"- Excel 兼容文件: {output_excel_file}")
print(f"- 最终行数: {len(df)}")
print("每列缺失值统计如下：")
print(df.isnull().sum())
