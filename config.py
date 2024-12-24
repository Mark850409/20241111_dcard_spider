# mysql
DRIVER = 'mysql+pymysql'
USERNAME = 'root'
PASSWORD = 'root'
HOST = 'markweb.idv.tw'
PORT = '9007'
DATABASE = 'restaurant'
OFFSET = 0
LIMIT = 500

# SQLALCHEMY
SQLALCHEMY_DATABASE_URI = "{}://{}:{}@{}:{}/{}?charset=utf8mb4".format(DRIVER, USERNAME, PASSWORD, HOST, PORT, DATABASE)
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 顏文字符號規則
EMOTICONS = {
    u":\u2010\)": "Happy face or smiley",
    u":\)": "Happy face or smiley",
    u":-\]": "Happy face or smiley",
    u":\]": "Happy face or smiley",
    u":-3": "Happy face smiley",
    u":3": "Happy face smiley",
    u":->": "Happy face smiley",
    u":>": "Happy face smiley",
    u"8-\)": "Happy face smiley",
    u":o\)": "Happy face smiley",
    u":-\}": "Happy face smiley",
    u":\}": "Happy face smiley",
    u":-\)": "Happy face smiley",
    u":c\)": "Happy face smiley",
    u":\^\)": "Happy face smiley",
    u"=\]": "Happy face smiley",
    u"=\)": "Happy face smiley",
    u":\u2010D": "Laughing, big grin or laugh with glasses",
    u":D": "Laughing, big grin or laugh with glasses",
    u"8\u2010D": "Laughing, big grin or laugh with glasses",
    u"8D": "Laughing, big grin or laugh with glasses",
    u"X\u2010D": "Laughing, big grin or laugh with glasses",
    u"XD": "Laughing, big grin or laugh with glasses",
    u"=D": "Laughing, big grin or laugh with glasses",
    u"=3": "Laughing, big grin or laugh with glasses",
    u"B\^D": "Laughing, big grin or laugh with glasses",
    u":-\)\)": "Very happy",
    u":\u2010\(": "Frown, sad, angry or pouting",
    u":-\(": "Frown, sad, angry or pouting",
    u":\(": "Frown, sad, angry or pouting",
    u":\u2010c": "Frown, sad, angry or pouting",
    u":c": "Frown, sad, angry or pouting",
    u":\u2010<": "Frown, sad, angry or pouting",
    u":<": "Frown, sad, angry or pouting",
    u":\u2010\[": "Frown, sad, angry or pouting",
    u":\[": "Frown, sad, angry or pouting",
    u":-\|\|": "Frown, sad, angry or pouting",
    u">:\[": "Frown, sad, angry or pouting",
    u":\{": "Frown, sad, angry or pouting",
    u":@": "Frown, sad, angry or pouting",
    u">:\(": "Frown, sad, angry or pouting",
    u":'\u2010\(": "Crying",
    u":'\(": "Crying",
    u":'\u2010\)": "Tears of happiness",
    u":'\)": "Tears of happiness",
    u"D\u2010'": "Horror",
    u"D:<": "Disgust",
    u"D:": "Sadness",
    u"D8": "Great dismay",
    u"D;": "Great dismay",
    u"D=": "Great dismay",
    u"DX": "Great dismay",
    u":\u2010O": "Surprise",
    u":O": "Surprise",
    u":\u2010o": "Surprise",
    u":o": "Surprise",
    u":-0": "Shock",
    u"8\u20100": "Yawn",
    u">:O": "Yawn",
    u":-\*": "Kiss",
    u":\*": "Kiss",
    u":X": "Kiss",
    u";\u2010\)": "Wink or smirk",
    u";\)": "Wink or smirk",
    u"\*-\)": "Wink or smirk",
    u"\*\)": "Wink or smirk",
    u";\u2010\]": "Wink or smirk",
    u";\]": "Wink or smirk",
    u";\^\)": "Wink or smirk",
    u":\u2010,": "Wink or smirk",
    u";D": "Wink or smirk",
    # More emoticons follow
}

# 文字雲需保留的詞性(中英對照)
POS_TRANSLATIONS = {
    "(n)": "名詞",
    "(v)": "動詞",
    "(a)": "形容詞",
    "(d)": "副詞",
    "(r)": "代詞",
    "(m)": "數量詞"
}

# 英文與符號規則
ENGILISH_PATTERN = r"[A-Za-z]"
CHINESE_PATTERN = r"[-\uff0c\u3002\uff01\uff1f\u3001\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u300c\u300d\ufe41\ufe42\u2014\u2026\u301c\ufe33.#\ufe5d\ufe5e\"'|>]+"
EMOJI_PATTERN = r"[\U0001F300-\U0001F5FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]|[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FA6F]|[\U0001FA70-\U0001FAFF]|[\U00002702-\U000027B0]|[\U000027A1]|[\U000025B6]+"
URL_PATTERN = r"https?://\S+|www\.\S+"

# 停用詞列表
stopwords = ['什麼','酒駕','覺得','知道','自己','就是','不是','可以','真的','沒有','根本','問題','大家','出來','應該','不要','不會','這樣','一個','還是','因為','所以','這種','我們','直接','如果','怎麼']

# 自定義辭典
CUSTOM_DICT_PATH = 'custom_dict.txt'

# 詞頻抓取前n筆
WORD_COUNTS = 10

# 保留的詞性
TARGET_POS = ["n", "v", "a"]  # 只保留名詞(n)、動詞(v)和形容詞(a)

# 文字雲詞頻最大值
TOP_N=10

# 頁數
OFFSET = 0
# 筆數
LIMIT = 10000


# 文字雲字型
FONTS="fonts/msjh.ttc"