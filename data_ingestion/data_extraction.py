import pandas as pd
from google_play_scraper import Sort, reviews
import os
import time
import random
import logging
from datetime import datetime

# تنظیمات لاگر
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper_main.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class DoctoralMultiScraper:
    def __init__(self, apps_config):
        self.apps_config = apps_config # لیستی از تنظیمات هر اپلیکیشن

    def _get_last_date(self, file_name):
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            if not df.empty:
                return pd.to_datetime(df['full_date']).max()
        return None

    def start_ingestion(self):
        for config in self.apps_config:
            app_id = config['id']
            country = config['country']
            lang = config['lang']
            target_count = config['target']
            
            file_name = f"data_{config['label']}_{lang}_{country}.csv"
            
            logging.info(f"🚀 شروع استخراج: {config['label']} | زبان: {lang} | سقف هدف: {target_count}")
            last_date = self._get_last_date(file_name)
            
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

                    if not result: break

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

                    logging.info(f"📦 {len(all_reviews)}/{target_count} رکورد برای {config['label']} ({lang}) دریافت شد.")
                    if not continuation_token: break
                    
                    time.sleep(random.uniform(4.0, 7.0))
                    
                except Exception as e:
                    logging.error(f"❌ خطا در اپ {app_id}: {e}")
                    break

            self._save(all_reviews, file_name)

    def _save(self, data, file_name):
        if not data: return
        new_df = pd.DataFrame(data)
        if os.path.exists(file_name):
            old_df = pd.read_csv(file_name)
            final_df = pd.concat([new_df, old_df], ignore_index=True).drop_duplicates(subset=['review_id'])
        else:
            final_df = new_df
        final_df.to_csv(file_name, index=False, encoding='utf-8')
        logging.info(f"✅ فایل {file_name} با موفقیت به‌روزرسانی شد. کل رکوردها: {len(final_df)}\n")

if __name__ == "__main__":
    # ترکیب دقیق زبان‌ها و کشورها به همراه سهمیه هر کدام
    target_configs = [
        # ChatGPT (5000 Total)
        {'id': 'com.openai.chatgpt', 'label': 'ChatGPT', 'country': 'us', 'lang': 'en', 'target': 4000},
        {'id': 'com.openai.chatgpt', 'label': 'ChatGPT', 'country': 'ir', 'lang': 'fa', 'target': 1000},
        
        # Gemini (5000 Total)
        {'id': 'com.google.android.apps.bard', 'label': 'Gemini', 'country': 'us', 'lang': 'en', 'target': 4000},
        {'id': 'com.google.android.apps.bard', 'label': 'Gemini', 'country': 'ir', 'lang': 'fa', 'target': 1000},
        
        # X (5000 Total)
        {'id': 'com.x.grok', 'label': 'Grok', 'country': 'us', 'lang': 'en', 'target': 4000},
        {'id': 'com.x.grok', 'label': 'Grok', 'country': 'ir', 'lang': 'fa', 'target': 1000},

        # Copilot (4000 Total)
        {'id': 'com.microsoft.copilot', 'label': 'Copilot', 'country': 'us', 'lang': 'en', 'target': 3200},
        {'id': 'com.microsoft.copilot', 'label': 'Copilot', 'country': 'ir', 'lang': 'fa', 'target': 800},
        
        # DeepSeek (3000 Total)
        {'id': 'com.deepseek.chat', 'label': 'DeepSeek', 'country': 'us', 'lang': 'en', 'target': 2400},
        {'id': 'com.deepseek.chat', 'label': 'DeepSeek', 'country': 'ir', 'lang': 'fa', 'target': 600},
        
        # Perplexity (3000 Total)
        {'id': 'ai.perplexity.app.android', 'label': 'Perplexity', 'country': 'us', 'lang': 'en', 'target': 2400},
        {'id': 'ai.perplexity.app.android', 'label': 'Perplexity', 'country': 'ir', 'lang': 'fa', 'target': 600}
    ]
    
    scraper = DoctoralMultiScraper(target_configs)
    
    # برای تست، می‌توانید مقادیر target در بالا را موقتاً روی 100 تنظیم کنید تا کل چرخه را ببینید.
    scraper.start_ingestion()