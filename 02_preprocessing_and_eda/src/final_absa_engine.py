import pandas as pd
import os
import glob
import re

def calculate_sentiment(text, score):
    """
    محاسبه بار احساسی با واژه‌نامه تمیز و غنی‌شده.
    """
    text = str(text).lower()
    
    # دیکشنری غنی‌شده با مرواریدهای استخراج شده از شبکه عصبی
    positive_words = ['good', 'great', 'awesome', 'accurate', 'insightful', 'fantastic', 'reliable', 'useful', 'عالی', 'خوب', 'بهترین', 'دقیق', 'درست', 'سریع', 'راحت', 'قشنگ', 'کاربردی', 'فوق العاده', 'راضی']
    negative_words = ['bad', 'slow', 'crash', 'bug', 'error', 'wrong', 'fake', 'laggy', 'buggy', 'crashed', 'sucks', 'بد', 'کند', 'هنگ', 'قطعی', 'فیلتر', 'ارور', 'باگ', 'دروغ', 'اشتباه', 'غلط', 'نمیشه', 'خراب', 'ضعیف', 'لگ', 'خطا']
    
    # استفاده از Regex برای تطابق دقیق‌تر کلمات
    pos_score = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', text) or (len(word) > 3 and word in text))
    neg_score = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', text) or (len(word) > 3 and word in text))

    if pos_score > neg_score:
        return 1
    elif neg_score > pos_score:
        return -1
    else:
        # اگر کلمه احساسی مشخصی نبود، از امتیاز ستاره کاربر کمک میگیریم
        return 1 if score >= 4 else -1

def run_final_absa(processed_dir):
    print("شروع محاسبه احساسات نهایی (ABSA) برای داده‌های واقعی...")
    
    final_files = glob.glob(os.path.join(processed_dir, 'ALL_*_BALANCED_ready_for_ml.csv'))
    
    for filepath in final_files:
        filename = os.path.basename(filepath)
        print(f"در حال پردازش: {filename}")
        
        df = pd.read_csv(filepath)
        
        # پر کردن مقادیر خالی احتمالی با صفر
        if 'Sentiment_Polarity' not in df.columns:
            df['Sentiment_Polarity'] = 0
        df['Sentiment_Polarity'] = df['Sentiment_Polarity'].fillna(0).astype(int)
        
        # اعمال تابع فقط روی رکوردهایی که بار احساسی ندارند (0 هستند)
        mask = df['Sentiment_Polarity'] == 0
        
        if mask.any():
            df.loc[mask, 'Sentiment_Polarity'] = df.loc[mask].apply(
                lambda row: calculate_sentiment(row['final_clean_content'], row['score']), 
                axis=1
            )
            
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
    print("\nعملیات با موفقیت پایان یافت! پازل داده‌ها ۱۰۰٪ تکمیل شد.")
    print("اکنون متغیرهای مستقل (X) و وابسته (Y) کاملاً آماده ورود به مدل Random Forest هستند.")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    run_final_absa(processed_dir)