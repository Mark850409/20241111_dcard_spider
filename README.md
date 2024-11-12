
# Dcard爬蟲

## 簡介

用來爬取Dcard所有看版文章、文章搜尋爬取、留言爬取

## 使用方式

請先安裝套件

```python
pip install -r requirement.txt
```

請先執行這個腳本，爬取`Dcrad`看板代號和名稱

```python
python dcard_type_spider.py
```

主程式：爬取文章標題、名稱、按讚數、留言數、分享數、摘要、時間
```python
python dcard_articles_spider.py
```


![](https://raw.githubusercontent.com/Mark850409/20241111_dcard_spider/master/images/202411121833001.png)


![](https://raw.githubusercontent.com/Mark850409/20241111_dcard_spider/master/images/202411121834347.png)


![](https://raw.githubusercontent.com/Mark850409/20241111_dcard_spider/master/images/202411121834573.png)

主程式執行完畢會取得`dcard_articles.csv`，其中`article_link`是我們爬取文章留言需要的欄位

接著執行此程式爬取Dcard留言

```python
python dcard_comments_spider.py
```

![](https://raw.githubusercontent.com/Mark850409/20241111_dcard_spider/master/images/202411121834541.png)