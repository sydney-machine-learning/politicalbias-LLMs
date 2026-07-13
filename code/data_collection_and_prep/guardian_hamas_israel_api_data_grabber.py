import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

# request an developer's API_KEY for free from the Guardian Network website, then replace the '===' below with the actual API_KEY
API_KEY = '==='
FROM_DATE = '2020-01-01'
TO_DATE = '2024-12-01'
PAGE_SIZE = 50

TAG_IDS = [
    'world/israel',
    'world/palestinian-territories',
    'world/hamas'
]

def html_to_text(html):
    if not isinstance(html, str):
        return ''
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator='\n', strip=True)

def fetch_guardian_by_tag(tag_id):
    all_articles = []
    base_url = 'https://content.guardianapis.com/search'
    params = {
        'from-date': FROM_DATE,
        'to-date': TO_DATE,
        'api-key': API_KEY,
        'show-fields': 'headline,body',
        'show-tags': 'keyword',
        'page-size': PAGE_SIZE,
        'order-by': 'newest',
        'tag': tag_id
    }

    resp = requests.get(base_url, params={**params, 'page': 1})
    if resp.status_code != 200:
        print(f"Tag {tag_id} 请求失败，状态码：{resp.status_code}")
        return []

    total_pages = resp.json()['response']['pages']
    print(f"Tag {tag_id} 总页数：{total_pages}")

    for page in range(1, total_pages + 1):
        print(f"📄 正在抓取 {tag_id} - 第 {page}/{total_pages} 页 ...")
        params['page'] = page
        resp = requests.get(base_url, params=params)
        if resp.status_code != 200:
            print(f"Tag {tag_id} 第{page}页请求失败，状态码：{resp.status_code}")
            time.sleep(5)
            continue

        results = resp.json()['response']['results']
        for item in results:
            article = {
                'title': item['fields'].get('headline', ''),
                'body_text': html_to_text(item['fields'].get('body', '')),
                'published_at': item.get('webPublicationDate', ''),
                'url': item.get('webUrl', '')
            }
            if article['body_text'].strip():
                all_articles.append(article)

        time.sleep(0.2)  # 防止触发速率限制

    return all_articles

# 抓取所有 tag 的数据
all_results = []
for tag_id in TAG_IDS:
    all_results.extend(fetch_guardian_by_tag(tag_id))

# 去重
df = pd.DataFrame(all_results).drop_duplicates(subset=['url'])

# 保留指定列
df = df[['title', 'body_text', 'published_at', 'url']]

# 保存为 CSV
df.to_csv('guardian_tag_filtered_3_keywords.csv', index=False, encoding='utf-8-sig')
print(f'✅ 已保存 {len(df)} 条记录为 guardian_tag_filtered_3_keywords.csv')
