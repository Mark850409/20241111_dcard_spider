import os
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from time import sleep
import requests
import undetected_chromedriver as uc
import shutil

from webdriver_manager.chrome import ChromeDriverManager

# 讀取 CSV 文件，包含文章連結
article_df = pd.read_csv('dcard_articles.csv')
article_links = article_df['article_link'].tolist()
article_titles = article_df['title'].tolist()

# 讓使用者輸入每篇文章要爬取的留言數
num_comments_to_scrape = int(input("請輸入每篇文章要爬取的留言數量："))

# 使用 undetected_chromedriver 初始化 Chrome 瀏覽器
driver = uc.Chrome(service=Service(ChromeDriverManager().install()))

# 將視窗最大化
driver.maximize_window()

comments = []
image_folder = "dcard_images"

# 移除並重新建立圖片儲存資料夾
if os.path.exists(image_folder):
    shutil.rmtree(image_folder)
os.makedirs(image_folder)

for index, link in enumerate(article_links):
    title = article_titles[index]  # 獲取文章標題
    try:
        # 前往指定的 Dcard 帖子頁面
        driver.get(link)

        # 等待頁面載入
        sleep(5)

        # 抓取文章內文
        try:
            content_element = driver.find_element(By.XPATH, "//div[contains(@class, 'd_cn_1t d_gk_31 d_7v_5 c1h57ajp')]//div[contains(@class, 'd_xa_34')]//span")
            article_content = content_element.text.strip()
        except:
            article_content = ""

        # 抓取文章內的所有圖片
        picture_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'd_gz_1nzmc2w c1gs4vo7')]//picture/source")
        image_urls = set()

        for picture in picture_elements:
            srcset = picture.get_attribute("srcset")
            if srcset:
                url = srcset.split(",")[0].strip().split(" ")[0]
                image_urls.add(url)

        # 下載所有圖片
        for idx, img_url in enumerate(image_urls):
            try:
                response = requests.get(img_url)
                if response.status_code == 200:
                    file_path = os.path.join(image_folder, f"{link.split('/')[-1]}_{idx}.jpg")
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
            except Exception as e:
                print(f"下載圖片 {img_url} 時發生錯誤：", e)

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
                        # 將學校名稱、留言內容、時間和文章內文添加到結果列表中
                        comments.append({
                            "title": title,
                            "article_link": link,
                            "school": school_name,
                            "content": article_content,
                            "text": comment_text,
                            "time": comment_time,

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
    df.columns = ['title', 'article_link', 'school','content','text', 'time']  # 設定欄位名稱
    df.to_csv('comments.csv', index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 編碼寫入 CSV
    print("已成功將留言儲存到 comments.csv 中")
else:
    print("沒有找到任何留言，無法寫入 CSV")
