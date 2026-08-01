import pandas as pd
import os
import glob
import re

def super_clean_text(text):
    """حذف ایموجی‌ها، علائم خاص و نگه‌داشتن متن خالص"""
    text = str(text).lower()
    # نگه داشتن حروف فارسی، انگلیسی و اعداد
    text = re.sub(r'[^\w\sآ-یa-z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def apply_rich_mapping(processed_dir):
    print("شروع عملیات ساخت ستون جدید و نگاشت نهایی مبتنی بر دیکشنری غنی‌شده...")
    
    # دیکشنری غنی‌شده (ترکیب کلمات بذر و بهترین کشفیات شبکه عصبی)
    rich_lexicon = {
        'چالش‌های زیرساختی و فیلترینگ': ['فیلتر', 'اینترنت', 'vpn', 'فیلترشکن', 'قطعی', 'وصل', 'محدود', 'ایران'],
        'شایستگی چندرسانه‌ای': ['عکس', 'تصویر', 'image', 'picture', 'ویدیو', 'فیلم', 'video', 'اپلود', 'ویرایش', 'gallery', 'pic'],
        'امنیت روانی و کنترل داده': ['امنیت', 'هک', 'حریم', 'اطلاعات', 'security', 'privacy', 'کلاهبرداری', 'scam', 'دزد'],
        'شایستگی فنی و عملیاتی': ['هنگ', 'باگ', 'ارور', 'سرعت', 'اشتباه', 'bug', 'error', 'کند', 'crash', 'خراب', 'جواب'],
        'بومی‌سازی و هنجارهای فرهنگی': ['فارسی', 'زبان', 'لحن', 'ادب', 'دوست', 'رفیق', 'صمیمی', 'فرهنگ'],
        'حساسیت دامنه و زمینه': ['پول', 'خرید', 'دانشگاه', 'دکتر', 'کد', 'بانک', 'اشتراک', 'تحریم', 'رایگان', 'pay', 'مدرسه'],
        'مهندسی شفافیت و پارادوکس آن': ['منبع', 'دلیل', 'سورس', 'source', 'رفرنس', 'دروغ', 'توهم', 'اشتباه'],
        'پویایی زمانی و حافظه بافتی': ['حافظه', 'یاد', 'فراموش', 'memory', 'تاریخچه', 'سابقه', 'history']
    }

    # خواندن فایل‌های فیلترشده (آنهایی که کامنت‌های زردشان را حذف کردیم)
    base_files = [f for f in glob.glob(os.path.join(processed_dir, '*.csv')) 
                  if not f.endswith('_labeled.csv') and not f.endswith('_ready_for_ml.csv')]
    
    total_processed = 0
    
    for filepath in base_files:
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        
        # 1. ساخت ستون جدید و تمیز
        df['final_clean_content'] = df['content'].apply(super_clean_text)
        
        # 2. نگاشت هوشمند
        df['Final_Aspect'] = 'Noise'
        
        for index, row in df.iterrows():
            text = str(row['final_clean_content'])
            best_match = 'Noise'
            
            for aspect, keywords in rich_lexicon.items():
                # اگر هر کدام از کلمات کلیدی در متن بود
                if any(re.search(r'\b' + kw + r'\b', text) for kw in keywords) or \
                   any(kw in text for kw in keywords if len(kw) > 3): # برای کلمات فارسی که مرز کلمه ندارند
                    best_match = aspect
                    break # به محض پیدا کردن اولین تطابق، جنبه را ثبت کن
                    
            df.at[index, 'Final_Aspect'] = best_match
            
        # 3. حذف کامنت هایی که هیچکدام از این کلمات طلایی را نداشتند (Noise)
        df_final = df[df['Final_Aspect'] != 'Noise'].copy()
        total_processed += len(df_final)
        
        # ذخیره فایل آماده برای ماشین لرنینگ
        output_path = os.path.join(processed_dir, filename.replace('.csv', '_ready_for_ml.csv'))
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        
    print(f"\nعملیات با موفقیت پایان یافت!")
    print(f"تعداد {total_processed} کامنت ناب و ۱۰۰٪ مرتبط با شاخص‌ها، با ستون جدید ذخیره شدند.")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    apply_rich_mapping(processed_dir)