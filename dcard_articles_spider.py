import csv
from time import sleep
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

if __name__ == '__main__':
    # 使用 undetected_chromedriver
    driver = uc.Chrome()
    wait = WebDriverWait(driver, 5)

    # 使用者輸入滾動次數
    max_scrolls = int(input("請輸入滾動次數: "))

    # 打開指定的網址
    driver.get('https://www.dcard.tw/f/talk')

    # 等待 5 秒讓頁面加載完成
    sleep(5)

    # 定義儲存文章資料的陣列
    all_articles = []
    seen_titles = set()  # 用於追蹤已經處理過的標題

    # 使用新的查找方法 (find_elements) 並提取標題、時間、摘要、按讚數、留言數、分享數、圖片網址、文章連結
    for _ in range(max_scrolls):
        articles = driver.find_elements(By.XPATH, '//div[@data-key]')
        for article in articles:
            try:
                title = article.find_element(By.XPATH, './/h2').text
                time_element = article.find_elements(By.XPATH, './/time')
                if not time_element:
                    continue  # 如果沒有時間元素，跳過該筆資料
                time = time_element[0].get_attribute('datetime')
                summary = article.find_element(By.XPATH, './/div[contains(@class, "d_dj_1gzgpud")]').text
                image_url = article.find_element(By.XPATH, './/img').get_attribute('src')
                article_link = article.find_element(By.XPATH, './/h2/a').get_attribute('href')

                if title not in seen_titles:
                    article_data = {
                        'title': title,
                        'time': time,
                        'summary': summary,
                        'image_url': image_url,
                        'article_link': article_link
                    }
                    seen_titles.add(title)
                    all_articles.append(article_data)
            except Exception as e:
                print(f'無法提取部分資料，跳過該筆資料: {e}')

        # 每次滾動 1000 像素
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 等待 5 秒以加載新內容
        sleep(5)

    # 將所有文章資料寫入 CSV 文件
    csv_file = 'dcard_articles.csv'
    csv_columns = ['title', 'time', 'summary', 'image_url', 'article_link']

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for article in all_articles:
                writer.writerow(article)
        print("已成功將留言儲存到 dcard_articles.csv 中")
    except IOError as e:
        print(f'寫入 CSV 文件時發生錯誤: {e}')

    # 關閉瀏覽器
    driver.quit()
