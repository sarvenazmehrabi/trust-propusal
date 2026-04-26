import pandas as pd
from google_play_scraper import Sort, reviews
import os
import time
import random
import logging
import socket
from datetime import datetime

# جلوگیری از فریز شدن کانکشن پایتون (تایم‌اوت 20 ثانیه‌ای)
socket.setdefaulttimeout(20)

# ==========================================
# تنظیمات ماشین زمان (Time Machine Settings)
# ==========================================
TARGET_YEAR = 2026
TARGET_MONTH = 4  # تنظیم روی ماه 4 (همان ماهی که داشتی می‌گرفتی)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "scraper_time_machine.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class DoctoralTimeMachineScraper:
    def __init__(self, apps_config):
        self.apps_config = apps_config

    def _get_last_date(self, csv_path):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                return pd.to_datetime(df['full_date']).max()
        return None

    def start_ingestion(self):
        logging.info(f"⏳ ماشین زمان فعال شد. هدف: سال {TARGET_YEAR}، ماه {TARGET_MONTH:02d}")
        
        for config in self.apps_config:
            app_id = config['id']
            country = config['country']
            lang = config['lang']
            target_count = config['target']
            
            filename_base = f"data_{config['label']}_{lang}_{country}_{TARGET_YEAR}_{TARGET_MONTH:02d}"
            csv_file = os.path.join(DATA_DIR, f"{filename_base}.csv")
            json_file = os.path.join(DATA_DIR, f"{filename_base}.json")
            
            logging.info(f"🚀 شروع استخراج: {config['label']} | زبان: {lang} | سقف هدف: {target_count}")
            last_date = self._get_last_date(csv_file)
            
            all_reviews = []
            continuation_token = None
            stop_signal = False
            skipped_count = 0

            while len(all_reviews) < target_count and not stop_signal:
                try:
                    result, continuation_token = reviews(
                        app_id, lang=lang, country=country,
                        sort=Sort.NEWEST, count=100, continuation_token=continuation_token
                    )

                    # سیستم تشخیص بلاک شدن IP و خواب 10 دقیقه‌ای
                    if not result: 
                        if len(all_reviews) == 0 and skipped_count == 0:
                            logging.warning("⚠️ لیست خالی! محدودیت IP اعمال شده. سیستم به مدت ۱۰ دقیقه (۶۰۰ ثانیه) متوقف می‌شود...")
                            time.sleep(600)
                            logging.info("🔄 پایان استراحت. تلاش مجدد برای ارتباط با سرور...")
                            continue # تلاش مجدد برای همین اپلیکیشن
                        else:
                            logging.info("🛑 به انتهای کامنت‌های موجود برای این اپلیکیشن رسیدیم.")
                            break

                    for rev in result:
                        rev_date = rev['at']
                        
                        if rev_date.year > TARGET_YEAR or (rev_date.year == TARGET_YEAR and rev_date.month > TARGET_MONTH):
                            skipped_count += 1
                            continue
                            
                        if rev_date.year < TARGET_YEAR or (rev_date.year == TARGET_YEAR and rev_date.month < TARGET_MONTH):
                            logging.info(f"🛑 به کامنت‌های قدیمی‌تر از {TARGET_YEAR}-{TARGET_MONTH:02d} رسیدیم. پایان این بخش.")
                            stop_signal = True
                            break

                        if last_date and rev_date <= last_date:
                            stop_signal = True
                            break

                        if rev_date.year == TARGET_YEAR and rev_date.month == TARGET_MONTH:
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
                            
                            if len(all_reviews) >= target_count:
                                stop_signal = True
                                break

                    if skipped_count > 0 and len(all_reviews) == 0:
                        if skipped_count % 1000 == 0:
                            logging.info(f"🕳️ در حال حفر کامنت‌های جدیدتر... ({skipped_count} کامنت رد شد. رسیدیم به: {result[-1]['at'].strftime('%Y-%m-%d')})")
                    elif len(all_reviews) > 0:
                        logging.info(f"📦 {len(all_reviews)}/{target_count} رکورد طلایی برای {config['label']} ({lang}) ذخیره شد.")
                    
                    if not continuation_token: break
                    
                    time.sleep(random.uniform(3.0, 6.0))
                    
                except Exception as e:
                    logging.error(f"❌ خطا در شبکه یا اپلیکیشن: {e}")
                    logging.info("⏳ به دلیل خطای شبکه، سیستم ۳ دقیقه استراحت می‌کند...")
                    time.sleep(180)
                    continue # در صورت خطای تایم‌اوت، 3 دقیقه صبر می‌کند و دوباره تلاش می‌کند

            self._save(all_reviews, csv_file, json_file)

    def _save(self, data, csv_file, json_file):
        if not data: return
        new_df = pd.DataFrame(data)
        
        if os.path.exists(csv_file):
            old_df = pd.read_csv(csv_file)
            final_df = pd.concat([new_df, old_df], ignore_index=True).drop_duplicates(subset=['review_id'])
        else:
            final_df = new_df
            
        final_df.to_csv(csv_file, index=False, encoding='utf-8')
        final_df.to_json(json_file, orient='records', force_ascii=False, indent=4)
        
        logging.info(f"✅ فایل‌های ماه {TARGET_MONTH} در مسیر data آپدیت شدند. کل رکوردها: {len(final_df)}\n")

if __name__ == "__main__":
    target_configs = [
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
    
    scraper = DoctoralTimeMachineScraper(target_configs)
    scraper.start_ingestion()