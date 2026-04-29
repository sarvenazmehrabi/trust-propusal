import os
import glob
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("🔄 در حال انتقال دیتای ماه 4 به سیستم یکپارچه (نسخه 9)...")

# پیدا کردن تمام فایل‌های ماه 4 که قبلاً گرفتیم
old_files = glob.glob(os.path.join(DATA_DIR, "data_*_2026_04.csv"))

if not old_files:
    print("❌ هیچ فایلی با پسوند 2026_04 پیدا نشد. مسیر را چک کنید.")
else:
    for old_file in old_files:
        filename = os.path.basename(old_file)
        
        # استخراج اسم چت‌بات، زبان و کشور از اسم فایل قدیمی
        # مثال: data_ChatGPT_en_us_2026_04.csv
        parts = filename.replace(".csv", "").split("_")
        app_label = parts[1]
        lang = parts[2]
        country = parts[3]
        
        # ساخت اسم جدید برای نسخه 9
        new_filename = f"balanced_data_{app_label}_{lang}_{country}.csv"
        new_filepath = os.path.join(DATA_DIR, new_filename)
        
        # کپی کردن فایل با اسم جدید (فایل قدیمی به عنوان بک‌آپ دست‌نخورده می‌ماند)
        shutil.copy(old_file, new_filepath)
        print(f"✅ تبدیل موفق: {filename} ➔ {new_filename}")

print("\n🚀 انتقال با موفقیت انجام شد! حالا می‌توانید اسکریپت data_extraction.py (نسخه 9) را اجرا کنید.")