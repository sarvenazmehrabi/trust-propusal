import pandas as pd
import os
import glob
import re

def calculate_sentiment(text):
    """
    محاسبه بار احساسی با رویکرد واژه‌نامه‌ای (Lexicon-based)
    مبتنی بر ادبیات کاربران ایرانی و جهانی
    """
    text = str(text).lower()
    
    # واژه‌نامه قطبیت مثبت (Positive Lexicon)
    positive_words = [
        'good', 'great', 'awesome', 'excellent', 'nice', 'best', 'love', 'fast', 'accurate', 'helpful',
        'عالی', 'خوب', 'بهترین', 'سریع', 'دقیق', 'مفید', 'تشکر', 'ممنون', 'مرسی', 'باحال', 'راضی', 'کمک'
    ]
    
    # واژه‌نامه قطبیت منفی (Negative Lexicon)
    negative_words = [
        'bad', 'slow', 'crash', 'bug', 'error', 'fake', 'lie', 'worst', 'terrible', 'stupid', 'not',
        'بد', 'کند', 'هنگ', 'قطعی', 'فیلتر', 'ارور', 'باگ', 'دروغ', 'مزخرف', 'چرت', 'نمیشه', 'خراب', 'ضعیف'
    ]
    
    # شمارش برخوردها (Hits)
    pos_score = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', text))
    neg_score = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', text))
    
    # برخی کلمات فارسی مرز کلمه‌ای \b ندارند، پس جستجوی ساده هم می‌کنیم
    if pos_score == 0 and neg_score == 0:
        pos_score = sum(1 for word in positive_words if word in text)
        neg_score = sum(1 for word in negative_words if word in text)

    # محاسبه قطبیت نهایی
    if pos_score > neg_score:
        return 1   # مثبت
    elif neg_score > pos_score:
        return -1  # منفی
    else:
        return 0   # خنثی

def run_sentiment_analysis(processed_dir):
    print("\nشروع فاز نهایی ABSA: محاسبه قطبیت احساسات (Sentiment Polarity)...")
    
    # پیدا کردن تمام فایل‌های لیبل‌خورده
    target_files = glob.glob(os.path.join(processed_dir, '*_labeled.csv'))
    
    total_processed = 0
    for filepath in target_files:
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        
        # ساخت ستون جدید برای احساسات
        df['Sentiment_Polarity'] = df['cleaned_content'].apply(calculate_sentiment)
        
        total_processed += len(df)
        
        # ذخیره نهایی با نام آماده برای مدل‌سازی (ready_for_ml)
        output_path = os.path.join(processed_dir, filename.replace('_labeled.csv', '_ready_for_ml.csv'))
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"فایل {filename} تحلیل احساسات شد.")

    print(f"\nپایان موفقیت‌آمیز فاز ABSA. تعداد {total_processed} کامنت دارای بار احساسی شدند.")
    print("داده‌های شما اکنون 100% برای ورود به مدل Random Forest آماده هستند!")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    run_sentiment_analysis(processed_dir)