from snownlp import SnowNLP
import pandas as pd


def analyze_sentiments_and_save_to_csv(input_csv_path, output_csv_path):
    try:
        # 讀取 CSV 檔案
        df = pd.read_csv(input_csv_path)
        
        if 'text' not in df.columns:
            raise ValueError("Input CSV does not contain 'text' column")

        sentiments = []
        sentiment_scores = []

        for text in df['text']:
            if isinstance(text, str):
                s = SnowNLP(text)
                sentiment_scores.append(round(s.sentiments, 2))  # 取到小數點兩位
                if s.sentiments > 0.6:
                    sentiments.append('正向情感')
                elif s.sentiments < 0.4:
                    sentiments.append('負向情感')
                else:
                    sentiments.append('中性情感')
            else:
                sentiments.append('中性情感')
                sentiment_scores.append(0.5)  # 預設中性分數

        # 將結果新增到 DataFrame
        df['sentiment'] = sentiments
        df['sentiment_score'] = sentiment_scores

        # 將更新後的 DataFrame 寫回 CSV
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Sentiment analysis results saved to {output_csv_path}")
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")

if __name__ == '__main__':
    # 示例用法
    input_csv = 'csv/comments.csv'
    output_csv = 'csv/comments_with_sentiments.csv'
    analyze_sentiments_and_save_to_csv(input_csv, output_csv)
