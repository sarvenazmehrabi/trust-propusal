import requests
import pandas as pd
import os
import time
import random
import logging

# ==========================================
# Paths and Configurations
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CafeBazaarScraper:
    def __init__(self, apps_config):
        self.apps_config = apps_config
        self.api_url = 'https://api.cafebazaar.ir/rest-v1/process/ReviewRequest'
        
        # Exact headers from your Chrome browser to bypass bot detection
        self.headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7',
            'content-type': 'application/json',
            'origin': 'https://cafebazaar.ir',
            'referer': 'https://cafebazaar.ir/',
            'sec-ch-ua': '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
        }
        
        # The magic tokens extracted from the cURL command
        self.client_id = "qmjrrz5qyt9vaewmc9q8tid87xf5qgvp"

    def start_ingestion(self):
        logging.info("🇮🇷 خزشگر کافه بازار (نسخه اسنایپر) فعال شد...")
        
        for config in self.apps_config:
            app_id = config['app_id']
            app_label = config['label']
            target_limit = config.get('target', 5000) # Aiming for 5k per app
            
            csv_file = os.path.join(DATA_DIR, f"native_data_{app_label}_cafebazaar.csv")
            logging.info(f"\n🚀 شروع استخراج: {app_label} | شناسه: {app_id} | هدف: {target_limit} رکورد")
            
            all_reviews = []
            start_idx = 0
            step = 20 # Matching the exact behavior of the website (fetching 20 comments per scroll)
            
            while len(all_reviews) < target_limit:
                end_idx = start_idx + step
                
                # Exact Payload structure matching the cURL
                payload = {
                    "properties": {
                        "language": 2,
                        "clientID": self.client_id,
                        "deviceID": self.client_id,
                        "clientVersion": "web"
                    },
                    "singleRequest": {
                        "reviewRequest": {
                            "packageName": app_id,
                            "start": start_idx,
                            "end": end_idx
                        }
                    }
                }
                
                try:
                    response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Debug log for the first request to ensure we bypassed the 400 error
                        if start_idx == 0:
                            logging.info(f"🔍 وضعیت سرور برای {app_label}: کد {data.get('properties', {}).get('statusCode')}")
                        
                        try:
                            reviews = data['singleReply']['reviewReply']['reviews']
                        except KeyError:
                            logging.warning("⚠️ به انتهای کامنت‌ها رسیدیم یا ساختار تغییر کرد.")
                            break
                            
                        if not reviews:
                            logging.info(f"🏁 تمام کامنت‌های در دسترس {app_label} به پایان رسید.")
                            break
                            
                        for rev in reviews:
                            # Skip completely empty comments
                            if not rev.get('comment'):
                                continue
                                
                            all_reviews.append({
                                "review_id": f"bazaar_{app_label}_{rev.get('id', random.randint(1000, 999999))}",
                                "user_name": rev.get('user', 'Unknown'),
                                "content": rev.get('comment', '').strip(),
                                "score": rev.get('rate', 0),
                                "full_date": rev.get('date', 'Unknown'),
                                "date": rev.get('date', 'Unknown'), # Farsi dates will be normalized in Phase 2
                                "time": "00:00:00",
                                "app_version": rev.get('appVersion', 'Unknown'),
                                "developer_reply": rev.get('reply', None),
                                "app_label": app_label,
                                "lang": 'fa'
                            })
                            
                        logging.info(f"📥 تعداد {len(all_reviews)} کامنت تا الان دریافت شد... (ردیف: {start_idx} تا {end_idx})")
                        
                        # Move to the next page
                        start_idx = end_idx
                        
                        # Sleep to mimic human scrolling
                        time.sleep(random.uniform(1.5, 3.0))
                        
                    else:
                        logging.error(f"❌ خطای HTTP {response.status_code}. مکث ۳۰ ثانیه‌ای...")
                        time.sleep(30)
                        
                except Exception as e:
                    logging.error(f"❌ خطای شبکه: {e}")
                    time.sleep(10)
                    
            # Final Save Process
            if all_reviews:
                # Trim the list if we fetched slightly more than the target limit
                if len(all_reviews) > target_limit:
                    all_reviews = all_reviews[:target_limit]
                    
                df = pd.DataFrame(all_reviews)
                # Drop exact duplicate texts from the same user to ensure data quality
                df.drop_duplicates(subset=['user_name', 'content'], inplace=True) 
                
                # utf-8-sig ensures proper Farsi display when opened in Microsoft Excel
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                logging.info(f"💾 تعداد {len(df)} رکورد تمیز برای {app_label} با موفقیت در فایل CSV ذخیره شد.")

if __name__ == "__main__":
    # Added Hooshang to the target list as well
    target_apps = [
        # {'app_id': 'ai.ivira.app', 'label': 'Vira', 'target': 10000},
        # {'app_id': 'app.gapgpt.twa', 'label': 'GapGPT', 'target': 10000},
        # {'app_id': 'ai.hooshang.app', 'label': 'Hooshang', 'target': 10000},
          {'app_id': 'ir.roboo.pwa', 'label': 'Roboo', 'target': 10000},
          {'app_id': 'com.smilinno.zigap', 'label': 'Zigap', 'target': 10000},
          {'app_id': 'com.tapsage.athena.domestic', 'label': 'Atena', 'target': 10000}
    ]
    
    scraper = CafeBazaarScraper(target_apps)
    scraper.start_ingestion()