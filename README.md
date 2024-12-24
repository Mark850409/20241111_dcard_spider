
# Dcard爬蟲+視覺化圖表

## 簡介

用來爬取Dcard所有看版文章、文章搜尋爬取、留言爬取+視覺化圖表

## 目錄
- [Dcard爬蟲+視覺化圖表](#dcard爬蟲視覺化圖表)
  - [簡介](#簡介)
  - [目錄](#目錄)
  - [使用方式](#使用方式)
    - [Dcard爬蟲步驟](#dcard爬蟲步驟)
      - [STEP1：請先將我的專案整包下載下來](#step1請先將我的專案整包下載下來)
      - [STEP2：安裝python套件](#step2安裝python套件)
      - [STEP3：爬取Dcard所有看板主程式](#step3爬取dcard所有看板主程式)
      - [STEP4：爬取Dcard文章主程式](#step4爬取dcard文章主程式)
      - [STEP5：爬取Dcard留言、圖片、文章連結、文章摘要、校名、留言時間主程式](#step5爬取dcard留言圖片文章連結文章摘要校名留言時間主程式)
      - [STEP6：留言轉換情感分析主程式](#step6留言轉換情感分析主程式)
    - [視覺化圖表執行步驟](#視覺化圖表執行步驟)
    - [文字雲視覺化圖表](#文字雲視覺化圖表)
      - [STEP1：執行以下python程式](#step1執行以下python程式)
      - [STEP2：執行成功後點擊此連結](#step2執行成功後點擊此連結)
    - [酒駕視覺化圖表](#酒駕視覺化圖表)
      - [STEP1：執行以下python程式](#step1執行以下python程式-1)
      - [STEP2：執行成功後點擊此連結](#step2執行成功後點擊此連結-1)

## 使用方式

> [!note] 
> 圖檔在程式執行時會自動保存在image專案目錄，可以不用再web介面手動下載

### Dcard爬蟲步驟

#### STEP1：請先將我的專案整包下載下來
git指令

```bash
git clone https://github.com/Mark850409/20241224_LangChainWithGradioBot.git
```

沒有git，進入此連結，點擊code → DownloadZIP

```
https://github.com/Mark850409/20241224_LangChainWithGradioBot.git
```

#### STEP2：安裝python套件

```python
pip install -r requirement.txt
```

#### STEP3：爬取Dcard所有看板主程式
請先執行這個腳本，爬取`Dcrad`看板代號和名稱

```python
python dcard_type_spider.py
```

#### STEP4：爬取Dcard文章主程式
主程式：爬取文章標題、名稱、按讚數、留言數、分享數、摘要、時間
```python
python dcard_articles_spider.py
```

#### STEP5：爬取Dcard留言、圖片、文章連結、文章摘要、校名、留言時間主程式

```python
python dcard_comments_spider.py
```

#### STEP6：留言轉換情感分析主程式

執行完成STEP5，會在`csv`目錄下產生`comments.csv`，請在執行這隻程式，執行完成後會在`csv`目錄下產生`comments_with_sentiments.csv`，這個檔案會包含`情感分析`欄位

```python
python sentiments.py
```

### 視覺化圖表執行步驟

### 文字雲視覺化圖表

#### STEP1：執行以下python程式

```python
python wordCloud_with_sentiments.py
```
#### STEP2：執行成功後點擊此連結
http://localhost:7864


### 酒駕視覺化圖表
#### STEP1：執行以下python程式

```python
python DrunkdrivingChart.py.py
```
#### STEP2：執行成功後點擊此連結
http://localhost:7863