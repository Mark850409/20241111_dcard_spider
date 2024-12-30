import jieba
import jieba.posseg as pseg
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import gradio as gr
from collections import Counter
import numpy as np
import matplotlib.font_manager as fm
import re
import config
from PIL import Image
import seaborn as sns
from matplotlib import colors
from config import FONTS
import random
from sqlalchemy import create_engine
import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from snownlp import SnowNLP

# 使用設定值
EMOTICONS = config.EMOTICONS
ENGILISH_PATTERN = config.ENGILISH_PATTERN
CHINESE_PATTERN = config.CHINESE_PATTERN
EMOJI_PATTERN = config.EMOJI_PATTERN
URL_PATTERN = config.URL_PATTERN
stopwords = config.stopwords
CUSTOM_DICT_PATH = config.CUSTOM_DICT_PATH
WORD_COUNTS = config.WORD_COUNTS
FONTS = config.FONTS
TARGET_POS = config.TARGET_POS  # 指定要保留的詞性列表


# 加載自定義辭典
def load_custom_dict():
    jieba.load_userdict(CUSTOM_DICT_PATH)  # 請將 'custom_dict.txt' 替換為你的自定義辭典檔案路徑


load_custom_dict()


def get_text_from_db():
    # 頁數
    offset = config.OFFSET
    # 筆數
    limit = config.LIMIT
    engine = create_engine(config.SQLALCHEMY_DATABASE_URI)

    try:
        gc.collect()  # 手動觸發垃圾回收
        query = f"SELECT text FROM google_maps_review_with_selenium_ollama"
        df = pd.read_sql(query, engine)
        if not df.empty:
            return ' '.join(df['text'].tolist())
    except Exception as e:
        print(f"資料庫讀取錯誤: {e}")
        return ""
    
# 讀取 CSV 檔案並提取 sentiment_score 欄位
def read_sentiments_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'sentiment_score' not in df.columns:
            raise ValueError("CSV 文件中未找到 'sentiment_score' 欄位")
        return df['sentiment_score']
    except Exception as e:
        print(f"讀取文件失敗: {str(e)}")
        return pd.Series(dtype='float64')

# 將情感分數轉換為類型（positive, neutral, negative）
def categorize_sentiments(sentiment_scores):
    categories = sentiment_scores.apply(
        lambda score: '正向情感' if score > 0.6 else '中性情感' if score < 0.4 else '負向情感'
    )
    return categories

# 讀取 Excel 文件中的情感分數並建立情感分析圖表
def analyze_sentiments_and_create_charts(file_path):
    sentiment_scores = read_sentiments_from_csv(file_path)
    if sentiment_scores.empty:
        print("未能讀取到有效的情感分數")
        return None, None

    # 將分數分類為類型
    sentiments = categorize_sentiments(sentiment_scores)
    # 繪製圖表
    sentiment_barchart = plot_sentiment_barchart(sentiments)
    sentiment_flowchart = plot_sentiment_flowchart(sentiments)

     # 新增直方圖和核密度估計圖
    histogram_path, kde_path = create_sentiment_distribution_plots(sentiment_scores)

    return sentiment_barchart, sentiment_flowchart, histogram_path, kde_path



# 讀取 Excel 文件中的內容欄位
def read_file(file):
    try:
        if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file, engine='openpyxl' if file.name.endswith('.xlsx') else 'xlrd')
            if 'text' in df.columns:
                return ' '.join(df['text'].dropna().tolist())
            else:
                return "Excel 文件中未找到 'text' 欄位"
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
            if 'text' in df.columns:
                return ' '.join(df['text'].dropna().tolist())
            else:
                return "CSV 文件中未找到 'text' 欄位"
        elif file.name.endswith('.txt'):
            import chardet
            with open(file.name, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            with open(file.name, 'r', encoding=encoding) as f:
                return f.read().strip()
        else:
            return ValueError("讀取文件失敗: 不支持的文件格式")
    except Exception as e:
        return f"讀取文件失敗: {str(e)}" if isinstance(e, Exception) else "讀取文件失敗: 發生未知錯誤"

# 建立情感分析長條圖
def plot_sentiment_barchart(sentiments):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.font_manager as fm

    plt.figure(figsize=(10, 6))
    sentiment_counts = sentiments.value_counts()
    sentiment_counts = sentiment_counts.reindex(['正向情感', '中性情感', '負向情感'], fill_value=0)

    ax = sentiment_counts.plot(
        kind='bar',
        color=sns.color_palette("pastel"),
        rot=0
    )

    ax.set_xlabel("情感類型\n(根據情感分數分類：正向>0.6, 負向<=0.4, 其餘為中性)", 
                 fontproperties=fm.FontProperties(fname=FONTS, size=12), labelpad=10)
    ax.set_ylabel("評論數量\n(屬於該情感類型的評論總數)", 
                 fontproperties=fm.FontProperties(fname=FONTS, size=12), labelpad=10)

    ax.set_xticklabels(sentiment_counts.index, 
                      fontproperties=fm.FontProperties(fname=FONTS, size=10))

    for i, v in enumerate(sentiment_counts):
        ax.text(
            i, v + 15,  # 上移15單位，避免被柱子遮住
            f'{int(v)}筆評論\n({v/sentiment_counts.sum()*100:.1f}%)',
            ha='center', va='bottom',
            fontproperties=fm.FontProperties(fname=FONTS, size=10)
        )

    # 計算主要情感傾向
    max_count_index = sentiment_counts.argmax()
    max_count = sentiment_counts.max()
    max_label = sentiment_counts.index[max_count_index]

    # 將黃色框移到新的紅框位置
    ax.text(
        max_count_index, max_count / 2,  # X 為柱狀圖的索引位置，Y 為柱子一半高度
        f'主要情感傾向：{max_label}\n共{int(max_count)}筆評論',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
        fontproperties=fm.FontProperties(fname=FONTS, size=12)
    )

    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    barchart_path = "image/sentiment_barchart.png"
    plt.savefig(barchart_path, format='png', dpi=300)
    plt.close()
    return barchart_path


# 建立情感分析圓餅圖
def plot_sentiment_flowchart(sentiments):
    plt.figure(figsize=(10, 6))
    sentiment_counts = sentiments.value_counts()
    sentiment_counts = sentiment_counts.reindex(['正向情感', '中性情感', '負向情感'], fill_value=0)

    total_reviews = sentiment_counts.sum()
    labels = sentiment_counts.index
    sizes = sentiment_counts.values
    patches, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=sns.color_palette("pastel"),
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*total_reviews)}筆)',
        startangle=140,
        textprops={'fontproperties': fm.FontProperties(fname=FONTS, size=10)},
        explode=(0.05,) * len(sizes)  # 動態設定 explode，與資料數量一致
    )

    plt.setp(autotexts, size=9, weight="bold")
    plt.title('情感分析圓餅圖\n(顯示各情感類型的占比分布)\n\n總評論數：{:,}筆'.format(int(total_reviews)), 
              fontproperties=fm.FontProperties(fname=FONTS, size=14), pad=20)

    legend_labels = [
        f'正向情感：情感分數 > 0.6',
        f'中性情感：0.4<=情感分數<=0.6',
        f'負向情感：情感分數 < 0.4'
    ]
    plt.legend(patches, legend_labels,
              title="情感分類標準",
              title_fontproperties=fm.FontProperties(fname=FONTS, size=12),  # 設定圖例標題字型
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              prop=fm.FontProperties(fname=FONTS, size=10))

    plt.axis('equal')
    plt.tight_layout()
    flowchart_path = "image/sentiment_flowchart.png"
    plt.savefig(flowchart_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    return flowchart_path

# 建立情感分析直方圖和核密度估計圖
def create_sentiment_distribution_plots(sentiment_scores):
    """
    創建情感分數的直方圖和核密度估計圖，並加入詳細的軸說明
    """
    # 確保數據不為空，且數據量足夠
    if sentiment_scores.empty or len(sentiment_scores) < 2:
        print("數據不足以生成核密度估計圖")
        return None
    
    # 創建直方圖
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(sentiment_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 添加更詳細的標題和軸標籤
    plt.title('情感分數分布直方圖\n(顯示各分數區間的評論數量)', 
             fontproperties=fm.FontProperties(fname=FONTS, size=14))
    plt.xlabel('情感分數\n(0表示最負面，1表示最正面)', 
             fontproperties=fm.FontProperties(fname=FONTS, size=12))
    plt.ylabel('評論數量\n(落在該分數區間的評論條數)', 
             fontproperties=fm.FontProperties(fname=FONTS, size=12))
    
    # 添加註解說明最高頻的區間
    max_count_index = np.argmax(counts)
    max_count_bin = (bins[max_count_index] + bins[max_count_index + 1]) / 2
    plt.annotate(
        f'最常見的情感分數區間\n{bins[max_count_index]:.2f}-{bins[max_count_index + 1]:.2f}\n共{int(counts[max_count_index])}條評論',
        xy=(max_count_bin, counts[max_count_index]),
        xytext=(10, 30),
        textcoords='offset points',
        ha='center',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        fontproperties=fm.FontProperties(fname=FONTS, size=10)
    )
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    histogram_path = "image/sentiment_histogram.png"
    plt.savefig(histogram_path, format='png', dpi=300)
    plt.close()
    
    # 創建核密度估計圖
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=sentiment_scores, color='skyblue', fill=True, alpha=0.5,clip=(0, 1), bw_adjust=0.5, kernel='epanechnikov')
    # 添加更詳細的標題和軸標籤
    plt.title('情感分數核密度估計圖\n(顯示情感分數的分布趨勢)', 
            fontproperties=fm.FontProperties(fname=FONTS, size=14))
    plt.xlabel('情感分數\n(0表示最負面，1表示最正面)', 
            fontproperties=fm.FontProperties(fname=FONTS, size=12))
    plt.ylabel('密度\n(情感分數在該處聚集的相對趨勢強度)', 
            fontproperties=fm.FontProperties(fname=FONTS, size=12))
    # 找出密度最高的點
    density = sns.kdeplot(data=sentiment_scores).get_lines()[0].get_data()
    max_density_index = np.argmax(density[1])
    max_density_score = density[0][max_density_index]
    # 添加註解說明最高密度點
    plt.annotate(
        f'情感分數最集中處\n分數約為{max_density_score:.2f}',
        xy=(max_density_score, density[1][max_density_index]),
        xytext=(10, 30),
        textcoords='offset points',
        ha='center',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        fontproperties=fm.FontProperties(fname=FONTS, size=10)
    )
    # 設置顯示範圍
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    kde_path = "image/sentiment_kde.png"
    plt.savefig(kde_path, format='png', dpi=300)
    plt.close()
    
    return histogram_path, kde_path


# 建立詞性標註與詞頻分析卡片
def create_word_cards(word_pos_freq):
    cards_html = """
    <h1 style='margin: 0; color: #fff; font-size: 1.5em;text-align:center;'>詞性標註與分析卡片</h1>
    <div style='
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 20px;
        width: 100%;
        padding: 20px;
    '>
    """
    for word_pos, freq in sorted(word_pos_freq.items(), key=lambda x: x[1], reverse=True)[:WORD_COUNTS]:
        word, pos = word_pos.split('/')
        pos_chinese = config.POS_TRANSLATIONS.get(pos, pos)  # 獲取詞性的中文註解
        cards_html += f"""
        <div style='
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        '>
            <h3 style='margin: 0; color: #333 !important; font-size: 1.5em;'>{word}</h3>
            <p style='margin: 10px 0; color: #333 !important;'>詞性: {pos_chinese}</p>
            <p style='margin: 10px 0; color: #333 !important;'>頻率: {freq}</p>
        </div>
        """
    cards_html += "</div>"
    return cards_html


# 建立詞性標註長條圖
def plot_word_frequency(word_pos_freq):
    plt.figure(figsize=(12, 10))

    words = []
    frequencies = []
    for word_pos, freq in sorted(word_pos_freq.items(), key=lambda x: x[1], reverse=True)[:WORD_COUNTS]:
        word, pos = word_pos.split('/')
        pos_chinese = config.POS_TRANSLATIONS.get(pos, pos)
        words.append(f"{word} 【{pos_chinese}】")
        frequencies.append(freq)

    y_pos = np.arange(len(words))
    colors = sns.color_palette("pastel")  # 隨機選擇莫蘭迪色系顏色
    bars = plt.barh(y_pos, frequencies, color=colors)

    # 改善圖表樣式
    plt.yticks(y_pos, words, fontproperties=fm.FontProperties(fname=FONTS, size=12))
    plt.xlabel('詞彙出現頻率', fontproperties=fm.FontProperties(fname=FONTS, size=12))
    plt.title(f'詞性標註與詞頻分析（前 {WORD_COUNTS} 個詞）',
              fontproperties=fm.FontProperties(fname=FONTS, size=14),
              pad=20)

    # 添加數值標籤
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}',
                 ha='left', va='center', fontsize=12)

    plt.gca().invert_yaxis()
    plt.tight_layout()

    barchart_path = "image/barchart.png"
    plt.savefig(barchart_path, format='png', dpi=300)
    plt.close()

    return barchart_path

# 建立詞性標註圓餅圖
def plot_pie_chart(word_pos_freq):
    plt.figure(figsize=(12, 10))

    labels = []
    sizes = []
    for word_pos, freq in sorted(word_pos_freq.items(), key=lambda x: x[1], reverse=True)[:WORD_COUNTS]:
        word, pos = word_pos.split('/')
        pos_chinese = config.POS_TRANSLATIONS.get(pos, pos)
        labels.append(f"{word}【{pos_chinese}】")
        sizes.append(freq)

    # 調整標籤位置及字型大小，避免重疊
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("pastel"),
        textprops={'fontproperties': fm.FontProperties(fname=FONTS, size=10)},  # 標籤字型大小
        labeldistance=1.1  # 調整標籤與圓心的距離
    )

    # 調整百分比文字的大小
    for autotext in autotexts:
        autotext.set_fontproperties(fm.FontProperties(fname=FONTS, size=9))

    plt.axis('equal')
    plt.tight_layout()

    piechart_path = "image/piechart.png"
    plt.savefig(piechart_path, format='png', dpi=300)
    plt.close()

    return piechart_path



# 生成文字雲
def process_text(input_text):
    if not input_text.strip():
        return None, None, "請輸入文字"
    
    input_texts = clean_text(input_text)
    final_texts = jieba_clean_text(input_texts)
    sentiments = analyze_sentiments_and_create_charts('csv/comments_with_sentiments.csv')
    sentiment_barchart, sentiment_flowchart, sentiment_histogram, sentiment_kde = sentiments
    words = pseg.cut(final_texts)
    word_list = []
    word_pos_freq = {}
    print("斷詞結果:", end=" ")
    count = 0
    for word, flag in words:
        if word.strip() and len(word) > 1 and flag in TARGET_POS:
            print(f"{word} ({flag})", end="; ")
            count += 1
            if count % 15 == 0:
                print()
            word_list.append(word)
            word_pos_key = f"{word}/({flag})"
            if word_pos_key in word_pos_freq:
                word_pos_freq[word_pos_key] += 1
            else:
                word_pos_freq[word_pos_key] = 1
    print("\n\n詞性標註與詞頻:")
    count = 0
    for word_pos, freq in sorted(word_pos_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"{word_pos}: {freq}", end="; ")
        count += 1
        if count % 15 == 0:
            print()

    if not word_list:
        return None, None, "未能識別出有效的詞"

    try:
        word_freq = Counter(word_list)

        # 读取mask图像
        mask = np.array(Image.open("image/python_logo.png"))  # 请将"mask_image.png"替换为你的mask图像路径

        # 使用莫蘭迪色系
        color_list = sns.color_palette("muted")

        # 调用
        colormap = colors.ListedColormap(color_list)

        # 生成文字雲
        wordcloud = WordCloud(
            font_path=FONTS,
            mask=mask,
            width=600,  # 調整寬度為更合適的顯示比例
            height=600,  # 調整高度為更合適的顯示比例
            margin=3,
            scale=10,
            background_color=None,  # 背景色設為白色
            colormap=colormap,  # 使用彩虹配色方案
            max_font_size=120,  # 設置最大字體大小
            min_font_size=10,  # 設置最小字體大小
            max_words=2000,  # 設置最大單詞數量
            min_word_length=0,
            include_numbers=False,
            repeat=False,
            prefer_horizontal=1,
            mode='RGBA',
            random_state=50,
            relative_scaling=1,
            font_step=4,
        ).generate_from_frequencies(word_freq)

        # 根據詞語頻率字典生成文字雲
        wordcloud.generate_from_frequencies(dict(word_freq))

        # 將詞頻列表轉換為字典
        word_freq = dict(word_freq)

        # 將詞頻重新排序
        sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

        # 強制設置前10名詞頻為最大值
        top_n = config.TOP_N
        max_freq = max(word_freq.values())
        adjusted_word_freq = {word: (max_freq if rank < top_n else freq) for rank, (word, freq) in
                              enumerate(sorted_word_freq.items())}

        # 根據詞語頻率字典生成文字雲
        wordcloud.generate_from_frequencies(adjusted_word_freq)

        # 讓文字雲跟著底圖顏色走
        image_colors = ImageColorGenerator(mask)

        plt.figure(figsize=(12, 10))
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)

        wordcloud_path = "image/wordcloud.png"
        plt.savefig(wordcloud_path, format='png', dpi=300)
        plt.close()

        barchart_path = plot_word_frequency(word_pos_freq)
        piechart_path = plot_pie_chart(word_pos_freq)
        word_cards_html = create_word_cards(word_pos_freq)

        tfidf_cards_html, tfidf_barchart_path, tfidf_piechart_path = calculate_tfidf(final_texts)
        return wordcloud_path, sentiment_barchart, sentiment_flowchart, sentiment_histogram, sentiment_kde,  barchart_path, piechart_path,  tfidf_barchart_path, tfidf_piechart_path, word_cards_html, tfidf_cards_html

    except Exception as e:
        return None, None, None, None, None, f"處理過程發生錯誤: {str(e)}"


# 計算 TF-IDF 並建立卡片
def calculate_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    tfidf_items = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:WORD_COUNTS]
    print("\n\nTF-IDF 分數:")
    for word, score in tfidf_items:
        print(f"{word}: {score:.4f}")
    plt.figure(figsize=(12, 10))
    words, scores = zip(*tfidf_items)
    y_pos = np.arange(len(words))
    colors = sns.color_palette("pastel")  # 隨機選擇莫蘭迪色系顏色
    plt.barh(y_pos, scores, color=colors)
    plt.gca().invert_yaxis()  # 確保高分在上方
    plt.yticks(y_pos, words, fontproperties=fm.FontProperties(fname=FONTS, size=12))
    plt.xlabel('TF-IDF 分數', fontproperties=fm.FontProperties(fname=FONTS, size=12))
    plt.title('TF-IDF 分數長條圖', fontproperties=fm.FontProperties(fname=FONTS, size=14))
    for index, value in enumerate(scores):
        plt.text(value, index, f'{value:.4f}', va='center', fontproperties=fm.FontProperties(fname=FONTS, size=10))
    plt.tight_layout(pad=3)
    tfidf_barchart_path = "image/tfidf_barchart.png"
    plt.savefig(tfidf_barchart_path, format='png', dpi=300)
    plt.close()

    # 繪製 TF-IDF 圓餅圖
    plt.figure(figsize=(12, 10))
    plt.pie(scores, labels=words, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"),
            textprops={'fontproperties': fm.FontProperties(fname=FONTS)})
    plt.axis('equal')
    plt.tight_layout()
    tfidf_piechart_path = "image/tfidf_piechart.png"
    plt.savefig(tfidf_piechart_path, format='png', dpi=300)
    plt.close()

    cards_html = """
        <h1 style='margin: 0; color: #fff; font-size: 1.5em;text-align:center;'>TF-IDF 分析卡片</h1>
        <div style='
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            width: 100%;
            padding: 20px;
        '>
        """
    for word, score in tfidf_items:
        cards_html += f"""
            <div style='
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            '>
                <h3 style='margin: 0; color: #333 !important; font-size: 1.5em;'>{word}</h3>
                <p style='margin: 10px 0; color: #333 !important;'>TF-IDF 分數: {score:.4f}</p>
            </div>
            """
    cards_html += "</div>"
    return cards_html, tfidf_barchart_path, tfidf_piechart_path


# 移除顏文字
def remove_emoticons(text):
    if text is None:
        return ""
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)


# 移除表情符號
def remove_emoji(text):
    if text is None:
        return ""
    # 正則表達式來匹配表情符號
    emoji_pattern = re.compile(
        EMOJI_PATTERN,
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


# 移除 URL
def remove_urls(text):
    if text is None:
        return ""
    # 正則表達式來匹配 URL
    url_pattern = re.compile(
        URL_PATTERN
    )
    return url_pattern.sub(r'', text)


# 移除英文字母
def remove_english_letters(text):
    # 正则表達式来匹配所有英文字母
    english_pattern = re.compile(ENGILISH_PATTERN)
    return english_pattern.sub(r'', text)


# 移除常用中文標點符號
def remove_chinese_comma(text):
    # 正则表達式来匹配所有英文字母
    chinese_pattern = re.compile(CHINESE_PATTERN)
    return chinese_pattern.sub(r'', text)


# 資料前處理
def clean_text(text):
    if text is None:
        return ""
    # 移除表情符號
    text = remove_emoji(text)
    # 移除 URL
    text = remove_urls(text)
    # 移除英文字母
    text = remove_english_letters(text)
    # 移除顏文字
    text = remove_emoticons(text)
    # 移除常用中文標點符號
    text = remove_chinese_comma(text)
    # 移除多餘的空白和前後空白
    text = re.sub(r'\s+', '', text).strip()
    text = text.replace('\n', '').replace('\r', '')
    return text


# 使用 jieba 進行斷詞，並過濾掉停用詞
def jieba_clean_text(text):
    words = jieba.cut(text, cut_all=False)
    filtered_words = [word for word in words if word not in stopwords]
    # print(filtered_words)
    return ' '.join(filtered_words)


def main():
    # 從資料庫獲取初始文本
    #initial_text = get_text_from_db()
    # 使用 Gradio 初始化模組
    with gr.Blocks(theme=gr.themes.Monochrome(), css=".gradio-container { text-align: center; }") as demo:
        gr.Markdown(
            """
            # 文字雲+情緒分析+詞頻分析生成器
            請輸入一段中文文本，或上傳文件 (Excel, TXT, CSV)，生成對應的文字雲並顯示詞性標註與詞頻分析的圖表和互動式卡片。
            """
        )
        with gr.Row():
            error_message = gr.Markdown("", visible=False)
        with gr.Row():
            excel_upload = gr.Files(label="僅支援上傳文件 (Excel, TXT, CSV)",
                                   file_types=[".xls", ".xlsx", ".txt", ".csv"], interactive=True)
            wordcloud_output = gr.Image(type="filepath", label="文字雲")
        with gr.Row():
            input_textbox = gr.Textbox(
                lines=15,
                label="輸入文本",
                placeholder="請在此輸入中文文本...",
                elem_classes="input-textbox",  # 添加自定義類別
                interactive=True,
                value=''
            )
        with gr.Row():
            submit_btn = gr.Button("提交")
            clear_btn = gr.Button("清除")
        with gr.Row():
            sentiment_barchart_output = gr.Image(label="情感分析長條圖")
            sentiment_piechart_output = gr.Image(label="情感分析圓餅圖")
        # 新增一行用於顯示直方圖和核密度估計圖
        with gr.Row():
            sentiment_histogram_output = gr.Image(label="情感分數分布直方圖")
            sentiment_kde_output = gr.Image(label="情感分數核密度估計圖")
        with gr.Row():
            barchart_output = gr.Image(type="filepath", label="詞性標註與詞頻分析長條圖")
            piechart_output = gr.Image(type="filepath", label="詞性標註頻率圓餅圖")
        with gr.Row():
            tfidf_barchart_output = gr.Image(type="filepath", label="TF-IDF 分數長條圖")
            tfidf_piechart_output = gr.Image(type="filepath", label="TF-IDF 分數圍餅圖")
        with gr.Row():
            word_cards_output = gr.HTML(label="詞性標註與詞頻分析卡片")
            tfidf_cards_output = gr.HTML(label="TF-IDF 分析卡片")

        def handle_input(input_text, files):
            combined_text = input_text
            if files is not None:
                for file in files:
                    file_text = read_file(file)
                    if isinstance(file_text, str):
                        combined_text += ' ' + file_text
            if not combined_text.strip():
                return gr.update(visible=True, value="請輸入文字或上傳檔案"), None, None, None, None, None, None, None, None, None, None, None, None
            return gr.update(visible=False, value=""), *process_text(combined_text)

        excel_upload.change(fn=lambda files: ' '.join(
            [read_file(file) for file in files if isinstance(read_file(file), str)]) if files else '',
                            inputs=excel_upload, outputs=input_textbox)
        submit_btn.click(
            fn=handle_input,
            inputs=[input_textbox, excel_upload],
            outputs=[
                error_message, wordcloud_output,
                sentiment_barchart_output, sentiment_piechart_output,
                sentiment_histogram_output, sentiment_kde_output,
                barchart_output, piechart_output,
                tfidf_barchart_output, tfidf_piechart_output,
                word_cards_output, tfidf_cards_output
            ]
        )
        clear_btn.click(
            fn=lambda: (gr.update(visible=False, value=""), '', None, None, None, None, None, None, None, None, None, None, None),
            inputs=[],
            outputs=[
                error_message, input_textbox, wordcloud_output,
                sentiment_barchart_output, sentiment_piechart_output,
                sentiment_histogram_output, sentiment_kde_output,
                barchart_output, piechart_output,
                tfidf_barchart_output, tfidf_piechart_output,
                word_cards_output, tfidf_cards_output
            ]
        )

    # 啟動介面
    demo.launch(share=True, server_name="0.0.0.0", server_port=7864)


# 啟動Gradio伺服器
if __name__ == '__main__':
    main()