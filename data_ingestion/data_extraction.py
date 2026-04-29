import pandas as pd
from google_play_scraper import Sort, reviews
import os
import time
import random
import logging
import socket
import pickle
from datetime import datetime

socket.setdefaulttimeout(20)

# ==========================================
# تنظیمات ماشین زمان و سهمیه‌بندی
# ==========================================
MONTHS_TO_GO_BACK = 6  # چند ماه به عقب برگردیم؟
START_YEAR = 2026      # سال شروع
START_MONTH = 4        # ماه شروع
# ==========================================

# ساخت لیست ماه‌های هدف (مثلاً از 2026-04 تا 6 ماه قبل)
target_months_keys = []
for i in range(MONTHS_TO_GO_BACK):
    y = START_YEAR
    m = START_MONTH - i
    if m <= 0:
        m += 12
        y -= 1
    target_months_keys.append(f"{y}-{m:02d}")

# قدیمی‌ترین ماه مجاز
OLDEST_MONTH = target_months_keys[-1]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TOKEN_DIR = os.path.join(BASE_DIR, "tokens")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TOKEN_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "scraper_quota_balancer.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class DoctoralQuotaScraper:
    def __init__(self, apps_config):
        self.apps_config = apps_config

    def _get_token_path(self, app_label, lang):
        return os.path.join(TOKEN_DIR, f"token_{app_label}_{lang}.pkl")

    def load_token(self, app_label, lang):
        path = self._get_token_path(app_label, lang)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_token(self, app_label, lang, token):
        path = self._get_token_path(app_label, lang)
        with open(path, 'wb') as f:
            pickle.dump(token, f)

    def start_ingestion(self):
        logging.info(f"🗓️ برنامه سهمیه‌بندی ۶ ماهه فعال شد. (از {target_months_keys[0]} تا {target_months_keys[-1]})")
        
        for config in self.apps_config:
            app_label = config['label']
            lang = config['lang']
            country = config['country']
            target_per_month = config['target']
            
            csv_file = os.path.join(DATA_DIR, f"balanced_data_{app_label}_{lang}_{country}.csv")
            
            logging.info(f"\n🚀 شروع/ادامه: {app_label} | سهمیه هر ماه: {target_per_month} رکورد")
            
            # خواندن آمار قبلی از CSV برای اینکه بداند کدام ماه‌ها پر شده‌اند
            current_counts = {m: 0 for m in target_months_keys}
            if os.path.exists(csv_file):
                old_df = pd.read_csv(csv_file)
                # استخراج ماه از ستون date و شمارش
                if not old_df.empty:
                    old_df['month_key'] = old_df['date'].str[:7]
                    counts = old_df['month_key'].value_counts().to_dict()
                    for k, v in counts.items():
                        if k in current_counts:
                            current_counts[k] = v
                logging.info(f"📊 وضعیت فعلی سهمیه‌ها در دیتابیس: {current_counts}")

            continuation_token = self.load_token(app_label, lang)
            if continuation_token:
                logging.info("⚡ توکن یافت شد! پرش مستقیم به آخرین نقطه...")

            stop_signal = False
            fast_forward_count = 0

            while not stop_signal:
                # چک کردن اینکه آیا کل ۶ ماه سهمیه‌شان پر شده است؟
                if all(count >= target_per_month for count in current_counts.values()):
                    logging.info("✅ سهمیه تمامی ۶ ماه برای این اپلیکیشن تکمیل شد. پایان.")
                    break

                try:
                    result, next_token = reviews(
                        config['id'], lang=lang, country=country,
                        sort=Sort.NEWEST, count=100, continuation_token=continuation_token
                    )

                    if not result: 
                        logging.warning("⚠️ لیست خالی! مسدودی IP. ده دقیقه استراحت...")
                        time.sleep(610)
                        continue

                    valid_reviews = []
                    saved_in_this_batch = 0

                    for rev in result:
                        rev_month = f"{rev['at'].year}-{rev['at'].month:02d}"
                        
                        # قانون 1: اگر رسیدیم به قبل از 6 ماه پیش -> توقف کامل اپلیکیشن
                        if rev_month < OLDEST_MONTH:
                            logging.info(f"🛑 رسیدیم به قبل از {OLDEST_MONTH}. پایان این اپلیکیشن.")
                            stop_signal = True
                            break

                        # قانون 2: بررسی سهمیه ماه
                        if rev_month in target_months_keys:
                            if current_counts[rev_month] < target_per_month:
                                # ظرفیت دارد -> ذخیره کن
                                valid_reviews.append({
                                    "review_id": rev['reviewId'],
                                    "user_name": rev['userName'],
                                    "content": rev['content'],
                                    "score": rev['score'],
                                    "full_date": rev['at'],
                                    "date": rev['at'].strftime('%Y-%m-%d'),
                                    "time": rev['at'].strftime('%H:%M:%S'),
                                    "app_version": rev.get('reviewCreatedVersion', 'Unknown'),
                                    "developer_reply": rev.get('replyContent', None),
                                    "app_label": app_label,
                                    "lang": lang
                                })
                                current_counts[rev_month] += 1
                                saved_in_this_batch += 1
                            else:
                                # ظرفیت این ماه پر شده -> نادیده بگیر و رد شو
                                fast_forward_count += 1

                    if valid_reviews:
                        self._append_to_csv(valid_reviews, csv_file)
                        
                    if not stop_signal and next_token:
                        self.save_token(app_label, lang, next_token)
                        continuation_token = next_token

                    if not next_token: break

                    # === دنده هوشمند و استراحت‌های بین‌راهی ===
                    if saved_in_this_batch == 0 and not stop_signal:
                        # fast_forward_count += 1
                        if fast_forward_count > 0 and (fast_forward_count % 100 == 0):
                            logging.info(f"⏩ دنده سریع: {fast_forward_count} کامنت اضافی رد شد. (رسیدیم به: {result[-1]['at'].strftime('%Y-%m-%d')})")
                        time.sleep(random.uniform(1.2, 2.5)) 
                    else:
                        logging.info(f"📦 وضعیت آپدیت شد: {current_counts}")
                        time.sleep(random.uniform(3.0, 5.5)) 
                    
                    # === اضافه کردن استراحت طولانی (Macro-Sleep) ===
                    # هر 2000 کامنت (ذخیره شده یا رد شده)، ربات 2 تا 4 دقیقه می‌خوابد
                    total_processed = sum(current_counts.values()) + fast_forward_count
                    if total_processed > 0 and total_processed % 2000 == 0:
                        coffee_break = random.uniform(120, 240)
                        logging.info(f"☕ زمان استراحت انسانی! ربات برای {int(coffee_break)} ثانیه متوقف می‌شود تا حساسیت گوگل کم شود...")
                        time.sleep(coffee_break)
                        logging.info("🚀 بازگشت به کار...")
                    
                except Exception as e:
                    logging.error(f"❌ خطای شبکه: {e} | سه دقیقه استراحت...")
                    time.sleep(180)
                    continue

    def _append_to_csv(self, new_data, csv_file):
        new_df = pd.DataFrame(new_data)
        if os.path.exists(csv_file):
            old_df = pd.read_csv(csv_file)
            final_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=['review_id'])
        else:
            final_df = new_df
        final_df.to_csv(csv_file, index=False, encoding='utf-8')

if __name__ == "__main__":
    target_configs = [
        # {'id': 'com.openai.chatgpt', 'label': 'ChatGPT', 'country': 'us', 'lang': 'en', 'target': 6000},
        # {'id': 'com.openai.chatgpt', 'label': 'ChatGPT', 'country': 'ir', 'lang': 'fa', 'target': 2000},
        # {'id': 'com.google.android.apps.bard', 'label': 'Gemini', 'country': 'us', 'lang': 'en', 'target': 6000},
        # {'id': 'com.google.android.apps.bard', 'label': 'Gemini', 'country': 'ir', 'lang': 'fa', 'target': 2000},
        {'id': 'ai.x.grok', 'label': 'Grok', 'country': 'us', 'lang': 'en', 'target': 6000},
        {'id': 'ai.x.grok', 'label': 'Grok', 'country': 'ir', 'lang': 'fa', 'target': 2000},
        {'id': 'com.microsoft.copilot', 'label': 'Copilot', 'country': 'us', 'lang': 'en', 'target': 5200},
        {'id': 'com.microsoft.copilot', 'label': 'Copilot', 'country': 'ir', 'lang': 'fa', 'target': 2800},
        {'id': 'com.deepseek.chat', 'label': 'DeepSeek', 'country': 'us', 'lang': 'en', 'target': 5200},
        {'id': 'com.deepseek.chat', 'label': 'DeepSeek', 'country': 'ir', 'lang': 'fa', 'target': 2800},
        {'id': 'ai.perplexity.app.android', 'label': 'Perplexity', 'country': 'us', 'lang': 'en', 'target': 5200},
        {'id': 'ai.perplexity.app.android', 'label': 'Perplexity', 'country': 'ir', 'lang': 'fa', 'target': 2800}
    ]
    
    scraper = DoctoralQuotaScraper(target_configs)
    scraper.start_ingestion()