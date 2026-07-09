import pandas as pd
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set DeepSeek API Base URL and API Key (Note: It is recommended to store the API key in environment variables to avoid leakage)
openai.api_base = "https://api.deepseek.com"
openai.api_key = ""

# Initialize client (New SDK must use `OpenAI()` instance)
client = openai.OpenAI(base_url=openai.api_base, api_key=openai.api_key)

df = pd.read_csv("guardian_russia_ukraine_total_filtered_upto_10000_tokens.csv")

# ========== Prompt Builder Function ==========
def build_prompt(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return f"""Read the content of the entire article and make a judgement whether this is
      'Left', 'Right' or 'Center'. Respond ONLY with one of the three exact words:
Left
Right
Center

Article:
{text}"""

def analyze_bias(i, content):
    prompt = build_prompt(content)  # Generate prompt using your function
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            stop=["\n", ".", ","]
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        result = f"[ERROR] {str(e)}"

    return i, result

# ✅ Process concurrently using ThreadPool
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(analyze_bias, i, content) for i, content in enumerate(df["content"])]

    for future in as_completed(futures):
        i, result = future.result()
        df.at[i, "bias"] = result
        print(f"Processed {i+1}/{len(df)}: {result}")

# ✅ Save processed result
df.to_csv("deepseek-RU-GUA.csv", index=False)
print("Saved to RU-GUA.csv")

def analyze_bias(i, content):
    prompt = build_prompt(content)  # Generate prompt using your function
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            stop=["\n", ".", ","]
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        result = f"[ERROR] {str(e)}"

    return i, result

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(analyze_bias, i, content) for i, content in enumerate(df["content"])]

    for future in as_completed(futures):
        i, result = future.result()
        df.at[i, "bias"] = result
        print(f"Processed {i+1}/{len(df)}: {result}")

# ✅ Save processed result
df.to_csv("BY-GUA.csv", index=False)
print("Saved to BY-GUA.csv")







