import pandas as pd
import os
import glob

def calculate_composite_trust(row):
    """
    تلفیق ستاره (رضایت سطحی) و لحن متن (احساس واقعی)
    برای ساخت شاخص ترکیبی اعتماد
    """
    score = row['score']
    polarity = row['Sentiment_Polarity']
    
    # اعتماد بالا (High Trust): کاربر هم امتیاز بالا داده و هم لحن متن مثبت است
    if score >= 4 and polarity == 1:
        return 1
    
    # بی‌اعتمادی (Low Trust): کاربر امتیاز پایین داده و لحن متن منفی است
    elif score <= 3 and polarity == -1:
        return 0
    
    # داده‌های متناقض / خاکستری (Contradictory): مثلاً ۵ ستاره داده ولی لحنش کاملاً منفی است
    # این داده‌ها برای آموزش ماشین‌لرنینگ سمی هستند و با مقدار 1- نشانه گذاری می شوند تا حذف گردند
    else:
        return -1

def apply_composite_trust(processed_dir):
    print("شروع عملیات مهندسی متغیر هدف (Y): ساخت شاخص ترکیبی اعتماد...")
    final_files = glob.glob(os.path.join(processed_dir, 'ALL_*_BALANCED_ready_for_ml.csv'))
    
    for filepath in final_files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        
        # اعمال منطق ترکیبی
        df['Composite_Trust'] = df.apply(calculate_composite_trust, axis=1)
        
        # حذف داده‌های متناقض که ماشین را گیج می‌کنند (حدود ۱۰ تا ۱۵ درصد داده‌ها)
        clean_df = df[df['Composite_Trust'] != -1].copy()
        
        print(f"\nفایل: {filename}")
        print(f"تعداد رکوردهای متناقض و حذف شده: {len(df) - len(clean_df)}")
        print(f"تعداد رکوردهای خالص و طلایی برای مدل‌سازی: {len(clean_df)}")
        
        # ذخیره روی همان فایل
        clean_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
    print("\nعملیات با موفقیت پایان یافت! متغیر هدف (Y) کاملاً منطبق بر اهداف رساله ایجاد شد.")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    apply_composite_trust(processed_dir)