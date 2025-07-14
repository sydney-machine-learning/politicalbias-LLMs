import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

API_KEY = '086ad3b3-a4f0-4d5d-9492-f4baa4d6a703'
FROM_DATE = '2020-01-01'
TO_DATE = '2024-12-01'
PAGE_SIZE = 50


TAG_IDS = [
    'world/ukraine',
    'world/russia',
    'world/ukraine-russia-war'
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
    
    # Grab the first page and the total pages
    resp = requests.get(base_url, params={**params, 'page': 1})
    if resp.status_code != 200:
        print(f"Tag {tag_id} Denied, status code：{resp.status_code}")
        return []
    
    total_pages = resp.json()['response']['pages']
    print(f"Tag {tag_id} Total Pages：{total_pages}")
    
    # traversal all pages
    for page in range(1, total_pages + 1):
        print(f"📄 Grabbing {tag_id} - Page Number {page}/{total_pages} ...")
        params['page'] = page
        
        resp = requests.get(base_url, params=params)
        if resp.status_code != 200:
            print(f"Tag {tag_id} Number {page} request denied，status code：{resp.status_code}")
            time.sleep(5)
            continue
        
        results = resp.json()['response']['results']
        for item in results:
            article = {
                'title': item['fields'].get('headline', ''),
                'body_text': html_to_text(item['fields'].get('body', '')),
                'published_at': item.get('webPublicationDate', ''),
                'url': item.get('webUrl', ''),
                'tag': tag_id  # plus the tag ID
            }
            
            if article['body_text'].strip():
                all_articles.append(article)
        
        time.sleep(0.2)  # prevent the limitation on speed
    
    return all_articles

# Grab all tagged data
all_results = []
for tag_id in TAG_IDS:
    print(f"🔍 Start grabbing tagged: {tag_id}")
    articles = fetch_guardian_by_tag(tag_id)
    all_results.extend(articles)
    print(f"✅ tag {tag_id} completed, received {len(articles)} articles")
    time.sleep(1)  # delays between tags

# Convert to DataFrame and remove duplicates
df = pd.DataFrame(all_results)
print(f"📊 prior to duplicate removal: {len(df)} articles")

# Removed Duplicates based on URL
df = df.drop_duplicates(subset=['url'])
print(f"📊 post duplicate removal: {len(df)} articles")

# rearrange columns
df = df[['title', 'body_text', 'published_at', 'url', 'tag']]

# Save as CSV
output_filename = 'guardian_russia_ukraine_articles.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f'✅ Save {len(df)} records as {output_filename}')

# show stats
print("\n📈 tag stats:")
print(df['tag'].value_counts())

print(f"\n📅 time span - from: {df['published_at'].min()} to {df['published_at'].max()}")
print(f"📄 average article length: {df['body_text'].str.len().mean():.0f} characters")