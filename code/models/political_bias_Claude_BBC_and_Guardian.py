import pandas as pd
import anthropic
import asyncio
import os
import time
import backoff
import logging
import tiktoken
import re
from typing import Tuple

# ========== 配置 ==========
INPUT_OUTPUT_PAIRS = [
    ("bbc_israel_2020_2024_advanced_filtered_filtered_upto_10000_tokens.csv", "bbc_israel_2020_2024_with_leaning.csv"),
    ("bbc_ukraine_2020_2024_advanced_filtered_filtered_upto_10000_tokens.csv", "bbc_ukraine_2020_2024_with_leaning.csv"),
    ("guardian_israel_hamas_total_filtered_upto_10000_tokens.csv", "guardian_israel_hamas_with_leaning.csv"),
    ("guardian_russia_ukraine_total_filtered_upto_10000_tokens.csv", "guardian_russia_ukraine_with_leaning.csv"),
]

TEXT_COLUMN = "content"
CONCURRENCY = 3
MODEL = "claude-3-haiku-20240307"
MAX_TOKENS_ALLOWED = 10000

# ========== 日志 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Claude 初始化 ==========
try:
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    logger.info("✅ Claude API client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Claude API client: {e}")
    raise

# ========== Token 编码器 ==========
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# ========== Prompt 构造 ==========
def build_prompt(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return f"""You are a political analyst. Read the following news content and classify its political leaning as one of the following: 
Left
Right
Centre

Respond ONLY with one of the three exact words.

Article:
{text}"""

# ========== 正则分类提取 ==========
def extract_leaning(reply: str, idx: int) -> str:
    reply_clean = reply.strip().lower()
    if re.search(r'\bleft\b', reply_clean):
        return "Left"
    elif re.search(r'\bright\b', reply_clean):
        return "Right"
    elif re.search(r'\b(center|centre|neutral)\b', reply_clean):
        return "Center"
    first_word = reply.strip().split()[0].lower()
    if first_word in ["left", "right", "center", "centre", "neutral"]:
        return {
            "left": "Left",
            "right": "Right"
        }.get(first_word, "Center")
    logger.warning(f"⚠️ Unexpected response at index {idx}: {reply}")
    return "Unknown"

# ========== 并发控制 ==========
sem = asyncio.Semaphore(CONCURRENCY)

# ========== Claude 请求 ==========
@backoff.on_exception(backoff.expo, (anthropic.RateLimitError, anthropic.APIConnectionError), max_tries=5, max_time=120)
async def call_claude_with_prompt(prompt_text: str) -> str:
    message = await client.messages.create(
        model=MODEL,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": prompt_text}]
    )
    await asyncio.sleep(0.8)
    return message.content[0].text.strip().replace("\n", "")

# ========== Claude 分类 ==========
async def classify_article(text: str, idx: int) -> Tuple[int, str]:
    async with sem:
        try:
            reply = await call_claude_with_prompt(build_prompt(text))
            logger.info(f"🧠 Claude reply {idx}: {reply}")
            return idx, extract_leaning(reply, idx)
        except Exception as e:
            logger.error(f"❌ Error at index {idx}: {e}, retrying with backup prompt")
            try:
                backup_prompt = f"Classify the political leaning of the article below as Left, Center, or Right. Respond ONLY with one of these three labels.\n\n{text}"
                reply = await call_claude_with_prompt(backup_prompt)
                logger.info(f"🔁 Backup reply {idx}: {reply}")
                return idx, extract_leaning(reply, idx)
            except Exception as e2:
                logger.error(f"❌ Backup attempt also failed at index {idx}: {e2}")
                return idx, "Unknown"

# ========== 主处理函数 ==========
async def run_batch_classification(input_file: str, output_file: str):
    logger.info(f"📂 Processing file: {input_file}")
    df = pd.read_csv(input_file)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")

    def within_token_limit(text): return len(encoding.encode(text)) <= MAX_TOKENS_ALLOWED
    df = df[df[TEXT_COLUMN].apply(within_token_limit)].copy()
    texts = df[TEXT_COLUMN].tolist()

    logger.info(f"🚀 Running Claude on {len(texts)} articles")
    tasks = [classify_article(text, i) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])

    df["leaning"] = [r[1] for r in results]
    df.to_csv(output_file, index=False)
    logger.info(f"✅ Saved output to: {output_file}")

    df_unknown = df[df["leaning"] == "Unknown"]
    if not df_unknown.empty:
        unknown_file = output_file.replace(".csv", "_unknown.csv")
        df_unknown.to_csv(unknown_file, index=False)
        logger.info(f"⚠️ Saved Unknowns to: {unknown_file}")

    logger.info(f"📊 {output_file} leaning counts:\n{df['leaning'].value_counts()}\n")

# ========== 执行 ==========
async def main():
    start = time.time()
    for input_file, output_file in INPUT_OUTPUT_PAIRS:
        await run_batch_classification(input_file, output_file)
    print(f"🕒 Total time elapsed: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
    asyncio.run(main())