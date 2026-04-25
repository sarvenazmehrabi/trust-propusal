import pandas as pd
from google_play_scraper import Sort, reviews
import os
import time
import random
import logging
from datetime import datetime

# پیدا کردن مسیر دقیق فایلی که در حال اجراست و ساخت پوشه data در کنار آن
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# تنظیمات لاگر
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "scraper_main.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class DoctoralMultiScraper:
    def __init__(self, apps_config):
        self.apps_config = apps_config

    def _get_last_date(self, csv_path):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                return pd.to_datetime(df['full_date']).max()
        return None

    def start_ingestion(self):
        for config in self.apps_config:
            app_id = config['id']
            country = config['country']
            lang = config['lang']
            target_count = config['target']
            
            # مسیردهی دقیق فایل‌ها داخل پوشه data
            csv_file = os.path.join(DATA_DIR, f"data_{config['label']}_{lang}_{country}.csv")
            json_file = os.path.join(DATA_DIR, f"data_{config['label']}_{lang}_{country}.json")
            
            logging.info(f"🚀 شروع استخراج: {config['label']} | زبان: {lang} | سقف هدف: {target_count}")
            last_date = self._get_last_date(csv_file)
            
            all_reviews = []
            continuation_token = None
            stop_signal = False

            while len(all_reviews) < target_count and not stop_signal:
                try:
                    batch_size = min(100, target_count - len(all_reviews))
                    result, continuation_token = reviews(
                        app_id, lang=lang, country=country,
                        sort=Sort.NEWEST, count=batch_size, continuation_token=continuation_token
                    )

                    # سیستم تشخیص بلاک شدن IP (Shadowban)
                    if not result: 
                        if len(all_reviews) == 0:
                            logging.warning("⚠️ گوگل لیستی برنگرداند! احتمالاً IP بلاک شده یا اپلیکیشن در این ریجن در دسترس نیست.")
                        break

                    for rev in result:
                        rev_date = rev['at']
                        if last_date and rev_date <= last_date:
                            stop_signal = True
                            break

                        all_reviews.append({
                            "review_id": rev['reviewId'],
                            "user_name": rev['userName'],
                            "content": rev['content'],
                            "score": rev['score'],
                            "full_date": rev_date,
                            "date": rev_date.strftime('%Y-%m-%d'),
                            "time": rev_date.strftime('%H:%M:%S'),
                            "app_version": rev.get('reviewCreatedVersion', 'Unknown'),
                            "developer_reply": rev.get('replyContent', None),
                            "app_label": config['label'],
                            "lang": lang
                        })

                    if len(all_reviews) > 0:
                        logging.info(f"📦 {len(all_reviews)}/{target_count} رکورد جدید برای {config['label']} ({lang}) دریافت شد.")
                    
                    if not continuation_token: break
                    
                    time.sleep(random.uniform(4.0, 7.0))
                    
                except Exception as e:
                    logging.error(f"❌ خطا در اپ {app_id}: {e}")
                    break

            self._save(all_reviews, csv_file, json_file)

    def _save(self, data, csv_file, json_file):
        if not data: return
        new_df = pd.DataFrame(data)
        
        # ذخیره CSV
        if os.path.exists(csv_file):
            old_df = pd.read_csv(csv_file)
            final_df = pd.concat([new_df, old_df], ignore_index=True).drop_duplicates(subset=['review_id'])
        else:
            final_df = new_df
            
        final_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # ذخیره JSON (جدید)
        final_df.to_json(json_file, orient='records', force_ascii=False, indent=4)
        
        logging.info(f"✅ فایل‌ها در مسیر پوشه data آپدیت شدند. کل رکوردها: {len(final_df)}\n")

if __name__ == "__main__":
    target_configs = [
        # لیست همان قبلی‌هاست (Grok و بقیه را به شکل دلخواه تنظیم کنید)
        {'id': 'com.openai.chatgpt', 'label': 'ChatGPT', 'country': 'us', 'lang': 'en', 'target': 4000},
        {'id': 'com.openai.chatgpt', 'label': 'ChatGPT', 'country': 'ir', 'lang': 'fa', 'target': 1000},
        
        {'id': 'com.google.android.apps.bard', 'label': 'Gemini', 'country': 'us', 'lang': 'en', 'target': 4000},
        {'id': 'com.google.android.apps.bard', 'label': 'Gemini', 'country': 'ir', 'lang': 'fa', 'target': 1000},

        {'id': 'ai.x.grok', 'label': 'Grok', 'country': 'us', 'lang': 'en', 'target': 4000},
        {'id': 'ai.x.grok', 'label': 'Grok', 'country': 'ir', 'lang': 'fa', 'target': 1000},
        
        {'id': 'com.microsoft.copilot', 'label': 'Copilot', 'country': 'us', 'lang': 'en', 'target': 3200},
        {'id': 'com.microsoft.copilot', 'label': 'Copilot', 'country': 'ir', 'lang': 'fa', 'target': 800},
        
        {'id': 'com.deepseek.chat', 'label': 'DeepSeek', 'country': 'us', 'lang': 'en', 'target': 2400},
        {'id': 'com.deepseek.chat', 'label': 'DeepSeek', 'country': 'ir', 'lang': 'fa', 'target': 600},
        
        {'id': 'ai.perplexity.app.android', 'label': 'Perplexity', 'country': 'us', 'lang': 'en', 'target': 2400},
        {'id': 'ai.perplexity.app.android', 'label': 'Perplexity', 'country': 'ir', 'lang': 'fa', 'target': 600}

        
    ]
    
    scraper = DoctoralMultiScraper(target_configs)
    scraper.start_ingestion()