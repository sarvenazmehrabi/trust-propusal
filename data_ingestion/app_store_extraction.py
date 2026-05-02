import pandas as pd
from app_store_scraper import AppStore
import os
import time
import random
import logging
from datetime import datetime

# ==========================================
# Time Machine & Quota Configuration
# ==========================================
MONTHS_TO_GO_BACK = 6
START_YEAR = 2026
START_MONTH = 4

# Generate target months list (e.g., ['2026-04', '2026-03', ...])
target_months_keys = []
for i in range(MONTHS_TO_GO_BACK):
    y = START_YEAR
    m = START_MONTH - i
    if m <= 0:
        m += 12
        y -= 1
    target_months_keys.append(f"{y}-{m:02d}")

# Define the absolute limit for backward scraping
OLDEST_MONTH = target_months_keys[-1]

# Directory setup for data and logs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "app_store_scraper.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class AppStoreQuotaScraper:
    def __init__(self, apps_config):
        self.apps_config = apps_config

    def start_ingestion(self):
        logging.info(f"🍏 خزشگر App Store فعال شد. بازه هدف: {target_months_keys[0]} تا {target_months_keys[-1]}")
        
        for config in self.apps_config:
            app_name = config['app_name']
            app_id = config['app_id']
            country = config['country']
            app_label = config['label']
            target_per_month = config['target']
            
            csv_file = os.path.join(DATA_DIR, f"balanced_data_{app_label}_ios_{country}.csv")
            logging.info(f"\n🚀 شروع استخراج: {app_label} (iOS) | ریجن: {country.upper()} | سهمیه: {target_per_month} رکورد")
            
            # Initialize quota tracking dictionary
            current_counts = {m: 0 for m in target_months_keys}
            
            # Load existing CSV to resume progress and prevent duplicates
            if os.path.exists(csv_file):
                old_df = pd.read_csv(csv_file)
                if not old_df.empty:
                    counts = old_df['date'].str[:7].value_counts().to_dict()
                    for k, v in counts.items():
                        if k in current_counts:
                            current_counts[k] = v
            
            logging.info(f"📊 وضعیت فعلی سهمیه‌ها (iOS): {current_counts}")
            
            # Skip if the 6-month quota is already fulfilled
            if all(count >= target_per_month for count in current_counts.values()):
                logging.info(f"✅ سهمیه ۶ ماهه {app_label} کامل است. پرش به اپلیکیشن بعدی...")
                continue

            try:
                logging.info(f"⏳ در حال واکشی کامنت‌ها از سرور اپل... (این مرحله ممکن است چند دقیقه طول بکشد)")
                
                # Initialize AppStore object
                app_instance = AppStore(country=country, app_name=app_name, app_id=app_id)
                app_instance.review(how_many=20000, sleep=random.randint(4, 7))
                
                fetched_reviews = app_instance.reviews
                logging.info(f"📥 تعداد کل کامنت‌های دریافت شده از سرور: {len(fetched_reviews)}")
                
                valid_reviews = []
                skipped_counts = {m: 0 for m in target_months_keys} # New numbering for dark data
                
                for rev in fetched_reviews:
                    rev_date = rev['date']
                    rev_month = f"{rev_date.year}-{rev_date.month:02d}"
                    
                    if rev_month < OLDEST_MONTH:
                        continue 
                        
                    if rev_month in target_months_keys:
                        if current_counts[rev_month] < target_per_month:
                            full_content = f"{rev.get('title', '')}. {rev.get('review', '')}".strip()
                            
                            valid_reviews.append({
                                "review_id": f"ios_{rev.get('userName', 'user')}_{rev_date.strftime('%Y%m%d%H%M%S')}",
                                "user_name": rev.get('userName', 'Unknown'),
                                "content": full_content,
                                "score": rev.get('rating', 0),
                                "full_date": rev_date,
                                "date": rev_date.strftime('%Y-%m-%d'),
                                "time": rev_date.strftime('%H:%M:%S'),
                                "app_version": rev.get('developerResponse', {}).get('id', 'Unknown'),
                                "developer_reply": rev.get('developerResponse', {}).get('body', None),
                                "app_label": app_label,
                                "lang": country 
                            })
                            current_counts[rev_month] += 1
                        else:
                            # This month's quota is full, so the record will be discarded.
                            skipped_counts[rev_month] += 1
                
                # Print dark data logs in a precise format for the analyst script
                for m, skipped in skipped_counts.items():
                    if skipped > 0:
                        logging.info(f"⏩ دنده سریع: {skipped} کامنت اضافی رد شد. (رسیدیم به: {m}-01)")
                
                if valid_reviews:
                    self._append_to_csv(valid_reviews, csv_file)
                    logging.info(f"💾 تعداد {len(valid_reviews)} رکورد معتبر جدید برای {app_label} ذخیره شد.")
                else:
                    logging.info(f"⚠️ رکورد جدیدی که در بازه زمانی و سهمیه ما بگنجد یافت نشد.")
                    
                logging.info(f"📦 وضعیت نهایی سهمیه‌ها پس از اجرا: {current_counts}")
                
            except Exception as e:
                logging.error(f"❌ خطای شبکه یا محدودیت اپل: {e}")

    def _append_to_csv(self, new_data, csv_file):
        """Helper method to append new records to CSV without duplicates."""
        new_df = pd.DataFrame(new_data)
        if os.path.exists(csv_file):
            old_df = pd.read_csv(csv_file)
            final_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=['review_id'])
        else:
            final_df = new_df
        final_df.to_csv(csv_file, index=False, encoding='utf-8')

if __name__ == "__main__":
    target_configs = [
        # Apple Store App IDs
        {'app_name': 'chatgpt', 'app_id': 1662168640, 'label': 'ChatGPT', 'country': 'us', 'target': 2500},
        {'app_name': 'microsoft-copilot', 'app_id': 6472538169, 'label': 'Copilot', 'country': 'us', 'target': 2500},
        {'app_name': 'perplexity-ask-anything', 'app_id': 1668000334, 'label': 'Perplexity', 'country': 'us', 'target': 2500},
        
        # Additional region to ensure enough data volume (UK store)
        {'app_name': 'chatgpt', 'app_id': 1662168640, 'label': 'ChatGPT', 'country': 'gb', 'target': 2500},
    ]
    
    scraper = AppStoreQuotaScraper(target_configs)
    scraper.start_ingestion()