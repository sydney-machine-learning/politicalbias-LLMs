import pandas as pd
import google.generativeai as genai
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set API Key
GOOGLE_API_KEY = "==="
genai.configure(api_key=GOOGLE_API_KEY)

# Create a model
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# prompt constructor
def build_prompt(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return f"""Read the content of the entire article and make a judgement whether this is
'Left', 'Right' or 'Center'. Respond ONLY with one of the three exact words:
Left
Right
Center

Article:
{text}"""

# Single-text classification function (with retry mechanism)
def classify_text(text: str) -> str:
    prompt = build_prompt(text)
    max_retries = 3
    sleep_time = 2

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[Retry {attempt}] Error: {e}")
            time.sleep(sleep_time * attempt)

    return "[Failed]"

# Batch classification function (concurrent processing)
def classify_texts_batch(text_list, max_workers=5):
    results = [None] * len(text_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(classify_text, text): i for i, text in enumerate(text_list)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                result = future.result()
                results[i] = result
                print(f"The classification of the {i+1}th article is complete: {result}")
            except Exception as e:
                print(f"Category failed for the {i+1}th article: {e}")
                results[i] = f"[Error] {e}"
    return results

# main part
if __name__ == "__main__":
    # Read data
    input_file = "==="
    df = pd.read_csv(input_file)

    # Select the text column
    if "content" not in df.columns:
        raise ValueError("The 'content' column cannot be found. Please check the column names in the CSV file.")

    texts = df["content"].astype(str).tolist()
    print(f"A total of {len(texts)} articles were read. Classification begins...\n")

    # Execution Classification
    df["bias"] = classify_texts_batch(texts)

    # Save results
    output_file = "==="
    df.to_csv(output_file, index=False)
    print(f"\n The classification is complete, and the results have been saved. {output_file}")
