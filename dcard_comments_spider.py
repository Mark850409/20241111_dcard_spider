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

# 初始化 Chrome 瀏覽器
try:
    driver = uc.Chrome(service=Service(ChromeDriverManager().install()))
    driver.maximize_window()
except Exception as e:
    print(f"初始化瀏覽器時發生錯誤：{e}")
    exit(1)

# 儲存留言和圖片資料
comments = []
image_folder = "dcard_images"

# 刪除並重新建立圖片儲存資料夾
if os.path.exists(image_folder):
    shutil.rmtree(image_folder)
os.makedirs(image_folder)

# 逐篇文章處理
for index, link in enumerate(article_links):
    title = article_titles[index]  # 獲取文章標題
    try:
        # 前往文章頁面
        driver.get(link)
        sleep(5)  # 等待頁面載入

        # 抓取文章內文
        try:
            content_element = driver.find_element(By.XPATH, "//div[contains(@class, 'd_cn_1t d_gk_31 d_7v_5 c1h57ajp')]//div[contains(@class, 'd_xa_34')]//span")
            article_content = content_element.text.strip()
        except Exception:
            article_content = ""

        # 抓取文章圖片
        picture_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'd_gz_1nzmc2w c1gs4vo7')]//picture/source")
        image_urls = {picture.get_attribute("srcset").split(",")[0].strip().split(" ")[0] for picture in picture_elements if picture.get_attribute("srcset")}

        # 下載圖片
        for idx, img_url in enumerate(image_urls):
            try:
                response = requests.get(img_url)
                if response.status_code == 200:
                    file_path = os.path.join(image_folder, f"{link.split('/')[-1]}_{idx}.jpg")
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
            except Exception as e:
                print(f"下載圖片 {img_url} 時發生錯誤：", e)

        # 滾動頁面抓取留言
        last_height = driver.execute_script("return document.body.scrollHeight")
        scraped_comments = 0

        while scraped_comments < num_comments_to_scrape:
            # 提取當前頁面留言
            comment_divs = driver.find_elements(By.XPATH, "//div[starts-with(@data-key, 'comment-')]")
            for div in comment_divs:
                if scraped_comments >= num_comments_to_scrape:
                    break
                try:
                    school_name = div.find_element(By.XPATH, ".//span[contains(@class, 't15mz9x4')]").text
                    comment_text = div.find_element(By.XPATH, ".//div[contains(@class, 'd_xa_34')]//span").text.strip()
                    comment_time = div.find_element(By.XPATH, ".//time").get_attribute("datetime")

                    if comment_text and not any(comment['text'] == comment_text and comment['article_link'] == link for comment in comments):
                        comments.append({
                            "title": title,
                            "article_link": link,
                            "school": school_name,
                            "content": article_content,
                            "text": comment_text,
                            "time": comment_time,
                        })
                        scraped_comments += 1
                except Exception:
                    continue

            # 滾動頁面加載新內容
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    except Exception as e:
        print(f"處理文章 {link} 時發生錯誤：", e)

# 確保關閉瀏覽器僅在程式完成後
try:
    if driver:
        driver.quit()
except Exception as e:
    print(f"關閉瀏覽器時發生錯誤：{e}")

# 將留言結果寫入 CSV
if comments:
    df = pd.DataFrame(comments)
    df.to_csv('comments.csv', index=False, encoding='utf-8-sig')
    print("已成功將留言儲存到 comments.csv 中")
else:
    print("沒有找到任何留言，無法寫入 CSV")
