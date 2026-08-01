import os
import glob
import pandas as pd

def clean_and_count(processed_dir):
    print("۱. شروع پاکسازی فایل‌های قدیمی و متداخل...")
    
    # پیدا کردن و حذف فایل‌های _labeled.csv که مربوط به مراحل قبل بودند
    old_labeled_files = glob.glob(os.path.join(processed_dir, '*_labeled.csv'))
    for f in old_labeled_files:
        try:
            os.remove(f)
        except Exception as e:
            pass
    print(f"تعداد {len(old_labeled_files)} فایل قدیمی و اضافی با موفقیت حذف شد.")
            
    print("\n۲. در حال محاسبه آمار دقیق داده‌های خالص...")
    # خواندن فایل‌های نهایی
    final_files = glob.glob(os.path.join(processed_dir, '*_ready_for_ml.csv'))
    
    global_count = 0
    native_count = 0
    
    for f in final_files:
        try:
            df = pd.read_csv(f)
            count = len(df)
            
            # تشخیص بومی یا جهانی بودن از روی نام فایل
            if 'native_data' in os.path.basename(f):
                native_count += count
            else:
                global_count += count
        except Exception as e:
            print(f"خطا در خواندن فایل {f}: {e}")
            
    print("\n==================================================")
    print("گزارش آماری داده‌های خالص (آماده برای دیتاسازی و مدل‌سازی):")
    print(f"تعداد کامنت‌های پلتفرم‌های جهانی (ChatGPT, Gemini و...): {global_count}")
    print(f"تعداد کامنت‌های پلتفرم‌های بومی (Atena, GapGPT و...): {native_count}")
    print(f"مجموع کل داده‌های طلایی: {global_count + native_count}")
    print("==================================================")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    clean_and_count(processed_dir)