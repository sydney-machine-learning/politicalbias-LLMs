import pandas as pd

# === 1. 读取 Excel 文件中的指定 sheet ===
input_file = "bbc_ur_bias_merge_result.xlsx"
df = pd.read_excel(input_file, sheet_name="Merged_Data")

# === 2. 保留 url 以 "http" 开头的行 ===
df = df[df["url"].astype(str).str.startswith("http")]

# === 3. 针对 url 去重（只保留第一条记录）===
df = df.drop_duplicates(subset=["url"], keep="first")

# === 4. 保存为新的 Excel 文件 ===
output_file = "bias_merge_result_filtered_dedup.xlsx"
df.to_excel(output_file, index=False)

print("✅ 已保存筛选 & 去重后的文件：", output_file)
print(f"保留的记录数：{len(df)} 行")
