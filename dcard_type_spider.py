import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc

# 目標網址
url = "https://www.dcard.tw/forum/popular"

# 設定 undetected ChromeDriver
options = webdriver.ChromeOptions()

# 初始化 undetected ChromeDriver
driver = uc.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 打開目標網址
driver.get(url)

# 將視窗最大化
driver.maximize_window()

# 等待頁面完全載入
time.sleep(5)

# 使用者輸入滾動次數
scroll_count = int(input("請輸入滾動次數: "))

# 滾動並提取看板列表項目
forums = []
for _ in range(scroll_count):
    # 滾動到頁面底部
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)

    # 等待新內容加載
    time.sleep(2)

    # 從特定類別中提取看板列表項目
    items = driver.find_elements(By.CSS_SELECTOR, ".d_2l_a.d_2l_a a[href^='/f']")
    for item in items:
        name = item.text.strip().split('\n')[1]  # 只取文本的第一部分
        href = item.get_attribute("href").split("/f/")[-1]
        if name and href and not any(f["alias"] == href for f in forums):
            forums.append({
                "name": name,
                "alias": href
            })

# 將資料寫入 JSON 文件
output = {"items": []}
for index, forum in enumerate(forums, start=1):
    output["items"].append({
        "id": index,
        "name": forum["name"],
        "alias": forum["alias"]
    })

with open("dcard_type.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("已成功將看板名稱寫入到 dcard_type.json")

# 關閉瀏覽器
driver.quit()
