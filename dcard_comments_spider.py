import pandas as pd
from selenium.webdriver.common.by import By
from time import sleep
import undetected_chromedriver as uc

# 讀取 CSV 文件，包含文章連結
article_df = pd.read_csv('dcard_articles.csv')
article_links = article_df['article_link'].tolist()

# 讓使用者輸入每篇文章要爬取的留言數
num_comments_to_scrape = int(input("請輸入每篇文章要爬取的留言數量："))

# 使用 undetected_chromedriver 初始化 Chrome 瀏覽器
driver = uc.Chrome()

# 將視窗最大化
driver.maximize_window()

comments = []

for link in article_links:
    try:
        # 前往指定的 Dcard 帖子頁面
        driver.get(link)

        # 等待頁面載入
        sleep(5)

        last_height = driver.execute_script("return document.body.scrollHeight")
        scraped_comments = 0

        while scraped_comments < num_comments_to_scrape:
            # 提取目前頁面上已加載的留言
            comment_divs = driver.find_elements(By.XPATH, "//div[starts-with(@data-key, 'comment-')]")

            for div in comment_divs:
                if scraped_comments >= num_comments_to_scrape:
                    break

                try:
                    # 提取學校名稱
                    school_name_span = div.find_element(By.XPATH, ".//span[contains(@class, 't15mz9x4')]")
                    school_name = school_name_span.text

                    # 提取留言內容
                    comment_span = div.find_element(By.XPATH, ".//div[contains(@class, 'd_xa_34')]//span")
                    comment_text = comment_span.text.strip()

                    # 如果留言為空，跳過該留言
                    if not comment_text:
                        continue

                    # 提取留言時間
                    time_element = div.find_element(By.XPATH, ".//time")
                    comment_time = time_element.get_attribute("datetime")

                    # 如果該留言已經存在於列表中，則跳過
                    if not any(comment['text'] == comment_text and comment['article_link'] == link for comment in comments):
                        # 將學校名稱、留言內容和時間添加到結果列表中
                        comments.append({
                            "article_link": link,
                            "school": school_name,
                            "text": comment_text,
                            "time": comment_time
                        })
                        scraped_comments += 1

                except Exception as e:
                    # 如果找不到某個元素，跳過該留言
                    continue

            # 滾動到頁面的底部
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # 等待 5 秒以加載新內容
            sleep(5)

            # 計算滾動後的新高度
            new_height = driver.execute_script("return document.body.scrollHeight")

            # 如果頁面高度沒有變化，則說明已經到達頁面底部
            if new_height == last_height:
                break
            last_height = new_height

    except Exception as e:
        print(f"在處理文章 {link} 時發生錯誤：", e)

# 關閉瀏覽器
driver.quit()

# 將結果寫入 CSV 檔案
if comments:
    df = pd.DataFrame(comments)
    df.columns = ['article_link', 'school', 'text', 'time']  # 設定欄位名稱為英文
    df.to_csv('comments.csv', index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 編碼寫入 CSV
    print("已成功將留言儲存到 comments.csv 中")
else:
    print("沒有找到任何留言，無法寫入 CSV")
