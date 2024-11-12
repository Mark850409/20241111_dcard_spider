import json
import csv
from time import sleep
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

# 讀取 JSON 檔案並解析其內容
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

if __name__ == '__main__':
    # 載入 dcard_type.json 檔案
    json_file_path = 'dcard_type.json'
    dcard_data = load_json(json_file_path)

    # 使用者選擇搜尋文章或輸入看板 ID
    search_type = input("請輸入要進行的操作 (s: 搜尋文章 / i: 輸入看板ID): ")
    if search_type.lower() == 'i':
        # 顯示可用的 ID 和名稱給使用者選擇
        for item in dcard_data['items']:
            print(f"看板ID: {item['id']}, 看板名稱: {item['name']}")

        # 使用者輸入要選擇的 ID
        selected_id = int(input("請輸入您要爬取的看板ID: "))
        selected_alias = None

        # 根據使用者輸入的 ID 找到對應的 alias
        for item in dcard_data['items']:
            if item['id'] == selected_id:
                selected_alias = item['alias']
                break

        if not selected_alias:
            print("無效的 ID，請重新運行程式並輸入有效的 ID。")
            exit()

        # 使用者選擇要爬取熱門還是最新文章
        article_type = input("請輸入要爬取的文章類型 (h: 熱門 / l: 最新): ")
        if article_type.lower() == "l":
            url = f'https://www.dcard.tw/f/{selected_alias}?tab=latest'
        else:
            url = f'https://www.dcard.tw/f/{selected_alias}'

    elif search_type.lower() == 's':
        # 使用者輸入搜尋關鍵字
        search_keyword = input("請輸入要搜尋的關鍵字: ")
        # 使用者選擇排序方式
        sort_type = input("請輸入排序方式 (t: 近期熱門 / l: 最新發布 / i: 最多互動 / r: 最相關): ")
        if sort_type.lower() == 't':
            sort_param = 'trending'
        elif sort_type.lower() == 'l':
            sort_param = 'latest'
        elif sort_type.lower() == 'i':
            sort_param = 'interaction'
        else:
            sort_param = 'relevance'
        url = f'https://www.dcard.tw/search?query={search_keyword}&sort={sort_param}'
    else:
        print("無效的選擇，請重新運行程式並輸入有效的選擇。")
        exit()

    # 使用 undetected_chromedriver
    driver = uc.Chrome()
    wait = WebDriverWait(driver, 5)

    # 使用者輸入滾動次數
    max_scrolls = int(input("請輸入滾動次數: "))

    # 打開指定的網址
    driver.get(url)

    # 將視窗最大化
    driver.maximize_window()

    # 等待 5 秒讓頁面加載完成
    sleep(5)

    # 定義儲存文章資料的陣列
    all_articles = []
    seen_titles = set()  # 用於追蹤已經處理過的標題

    # 使用新的查找方法 (find_elements) 並提取標題、時間、摘要、圖片網址、文章連結
    for _ in range(max_scrolls):
        articles = driver.find_elements(By.XPATH, '//div[@data-key]')
        for article in articles:
            try:
                title = article.find_element(By.XPATH, './/h2').text
                time_element = article.find_elements(By.XPATH, './/time')
                if not time_element:
                    continue  # 如果沒有時間元素，跳過該筆資料
                time = time_element[0].get_attribute('datetime')
                summary = article.find_element(By.XPATH, './/div[contains(@class, "d_d8_1nn1f8g")]').text

                # 如果沒有圖片網址，跳過該筆資料
                try:
                    image_url = article.find_element(By.XPATH, './/img').get_attribute('src')
                except Exception as e:
                    continue

                article_link = article.find_element(By.XPATH, './/h2/a').get_attribute('href')

                # 如果是看板ID，提取學校名稱，否則跳過
                school_name = None
                personal_alias = None
                if search_type.lower() == 'i':
                    try:
                        # 嘗試抓取學校名稱
                        school_name = article.find_element(By.XPATH, './/div[contains(@class, "d_xa_2b d_tx_2c d_lc_1u l814vja")]').text
                    except:
                        pass
                    try:
                        # 嘗試抓取個人卡稱
                        personal_alias = article.find_element(By.XPATH, './/div[contains(@class, "d_a5_3t d_h_1q d_ju_1s t8ve51q")]').text
                    except:
                        pass

                    # 跳過學校名稱包含 "客服小天使" 的文章
                    if school_name and "客服小天使" in school_name:
                        continue

                like_count = article.find_element(By.XPATH, './/div[@class="d_a5_1p d_h_1q d_mh_1t d_de_24 d_7v_7 d_d8_2s d_cn_2i d_gk_27 d_dz43bx_1s fnja3xi"]').text
                comment_count = article.find_element(By.XPATH, './/div[@class="d_a5_1p d_h_1q d_mh_1t d_de_24 d_7v_7 d_d8_2s d_cn_2i d_gk_27 d_dz43bx_1s fnja3xi"]/following-sibling::div').text
                save_count = article.find_element(By.XPATH, './/div[@class="d_ng_1z d_y6_1w ayeirqn"]').text

                # 強迫將非數值型態轉換為 0
                like_count = int(like_count) if like_count.isdigit() else 0
                comment_count = int(comment_count) if comment_count.isdigit() else 0
                save_count = int(save_count) if save_count.isdigit() else 0

                if title not in seen_titles:
                    article_data = {
                        'title': title,
                        'time': time,
                        'summary': summary,
                        'image_url': image_url,
                        'article_link': article_link,
                        'like_count': like_count,
                        'comment_count': comment_count,
                        'save_count': save_count
                    }
                    if school_name:
                        article_data['school_name'] = school_name
                    if personal_alias:
                        article_data['personal_alias'] = personal_alias
                    seen_titles.add(title)
                    all_articles.append(article_data)
            except Exception as e:
                print(f'無法提取部分資料，跳過該筆資料: {e}')

        # 每次滾動 1000 像素
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 等待 5 秒以加載新內容
        sleep(5)

    # 根據是否為搜尋文章，調整 CSV 欄位
    if search_type.lower() == 'i':
        csv_columns = ['title', 'time', 'summary', 'image_url', 'article_link', 'school_name', 'personal_alias', 'like_count', 'comment_count', 'save_count']
    else:
        csv_columns = ['title', 'time', 'summary', 'image_url', 'article_link', 'like_count', 'comment_count', 'save_count']

    # 將所有文章資料寫入 CSV 文件
    csv_file = 'dcard_articles.csv'

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for article in all_articles:
                # 根據欄位是否存在來寫入資料
                filtered_article = {key: article.get(key, '') for key in csv_columns}
                writer.writerow(filtered_article)
        print("已成功將留言儲存到 dcard_articles.csv 中")
    except IOError as e:
        print(f'寫入 CSV 文件時發生錯誤: {e}')

    # 關閉瀏覽器
    driver.quit()
