import os
import pandas as pd
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
from collections import Counter

# ==================== 模型路径 ====================
model_path = "./news_model_output/checkpoint-1182"

# ==================== 加载模型和分词器 ====================
print("🔍 Loading model...")
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

# ==================== 标签映射（根据训练模型实际输出） ====================
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# ==================== 分段函数：每段不超过512 tokens ====================
def split_into_chunks(text, max_tokens=512, stride=256):
    input_ids = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(input_ids), stride):
        chunk_ids = input_ids[i:i+max_tokens]
        if not chunk_ids:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if i + max_tokens >= len(input_ids):
            break
    return chunks

# ==================== Chunk-Vote 情感预测 ====================
def predict_sentiment_chunked(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return "error"

        chunks = split_into_chunks(text)
        if not chunks:
            return "error"

        labels = []
        for chunk in chunks:
            result = classifier(chunk, truncation=True, max_length=512)
            label = result[0]["label"]
            labels.append(label_map.get(label, "unknown"))

        # 多数投票
        valid_labels = [l for l in labels if l != "unknown"]
        if valid_labels:
            final = Counter(valid_labels).most_common(1)[0][0]
            return final
        else:
            return "error"
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return "error"

# ==================== 处理函数：加入一列predicted_sentiment ====================
def process_file(input_csv_path, output_csv_path):
    print(f"\n📄 Processing: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    if "content" not in df.columns:
        print("❌ No 'content' column found. Skipping.")
        return

    df = df[df["content"].notna() & (df["content"].str.strip() != "")].copy()

    tqdm.pandas(desc="🔍 Full-text Sentiment")
    df["predicted_sentiment"] = df["content"].progress_apply(predict_sentiment_chunked)

    df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved: {output_csv_path}")

# ==================== 待处理的文件 ====================
files = [
    ("bbc_bias_labeled_ip.csv", "bbc_isreal_with_sentiment.csv"),
    ("bbc_bias_labeled_ur.csv", "bbc_ukraine_with_sentiment.csv"),
    ("guardian_bias_labeled_ip.csv", "guardian_isreal_with_sentiment.csv"),
    ("guardian_bias_labeled_ur.csv", "guardian_ukraine_with_sentiment.csv"),
]

for input_file, output_file in files:
    if os.path.exists(input_file):
        process_file(input_file, output_file)
    else:
        print(f"⚠️ Missing file: {input_file}")

print("\n🎉 Full-text sentiment analysis complete!")
