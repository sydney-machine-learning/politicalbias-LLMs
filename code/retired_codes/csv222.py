import pandas as pd
from pathlib import Path

guardian_ip_xlsx = r"C:\Users\chenh\PycharmProjects\pythonProject4\final_result\merge\guardian_ip_models_merged_http_filtered_cleaned.xlsx"
guardian_ru_xlsx = r"C:\Users\chenh\PycharmProjects\pythonProject4\final_result\merge\guardian_ur2_models_merged_http_filtered_cleaned.xlsx"

guardian_ip_csv = Path(guardian_ip_xlsx).with_suffix(".csv")
guardian_ru_csv = Path(guardian_ru_xlsx).with_suffix(".csv")

df_ip = pd.read_excel(guardian_ip_xlsx, sheet_name=0, dtype="string")
df_ip.to_csv(guardian_ip_csv, index=False, encoding="utf-8-sig")

df_ru = pd.read_excel(guardian_ru_xlsx, sheet_name=0, dtype="string")
df_ru.to_csv(guardian_ru_csv, index=False, encoding="utf-8-sig")

print("转换完成：")
print("以哈 Guardian CSV ->", guardian_ip_csv)
print("俄乌 Guardian CSV ->", guardian_ru_csv)