import pandas as pd
import os
import re
from snownlp import SnowNLP
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 載入 .env 檔案中的環境變數
load_dotenv()

# 定義文本清理函數
def clean_text(text):
    """
    清理文本，移除換行符號、多餘空白及表情符號。
    """
    # 移除換行符號和多餘空白
    text = re.sub(r'[\r\n]+', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text)
    # 移除表情符號和非文字字符
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text, flags=re.UNICODE)
    return text

# 定義情感分析函數（使用 SnowNLP）
def analyze_sentiment_with_snownlp(texts):
    results = []
    for text in texts:
        if not text.strip():
            # 如果文本為空或只有空白，跳過處理
            results.append((0.5, '中性情感'))
            print("文本為空或無效，默認為中性情感。")
            continue
        try:
            s = SnowNLP(text)
            sentiment_score = round(s.sentiments, 2)  # 情感分數（0: 負面, 1: 正面）
            if sentiment_score > 0.6:
                sentiment = '正向情感'
            elif sentiment_score < 0.5:
                sentiment = '負向情感'
            else:
                sentiment = '中性情感'
            print(f"文本: {text} -> 情感分數: {sentiment_score}, 情感分類: {sentiment}")
            results.append((sentiment_score, sentiment))
        except Exception as e:
            print(f"處理文本時發生錯誤: {text}, 錯誤: {e}")
            results.append((0.5, '中性情感'))  # 默認為中性情感
    return results

def process_and_save(input_csv_path, output_csv_path):
    try:
        df = pd.read_csv(input_csv_path)

        if 'text' not in df.columns:
            raise ValueError("Input CSV does not contain 'text' column")

        # 清理文本
        df['text'] = df['text'].fillna('').apply(clean_text)

        # 執行情感分析
        texts = df['text'].tolist()
        sentiment_results = analyze_sentiment_with_snownlp(texts)
        df[['sentiment_score', 'sentiment']] = pd.DataFrame(sentiment_results, index=df.index)

        # 將結果保存到 CSV
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to {output_csv_path}")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == '__main__':
    # 示例用法
    input_csv = 'csv/comments.csv'
    output_csv = 'csv/comments_with_sentiments.csv'

    # 統一處理情感分析
    process_and_save(input_csv, output_csv)
