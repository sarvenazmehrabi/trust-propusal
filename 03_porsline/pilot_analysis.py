import pandas as pd
import pingouin as pg
import numpy as np

# 1. خواندن فایل اکسل خروجی پرس‌لاین
file_name = "data_porsline.xlsx"
df = pd.read_excel(file_name)

# حذف دو سطر اول که توضیحات اضافی پرس‌لاین هستند (در صورت نیاز)
df_clean = df.iloc[2:].copy()

# 2. دیکشنری تبدیل طیف لیکرت به عدد
likert_mapping = {
    'کاملاً مخالفم': 1,
    'مخالفم': 2,
    'تا حدی مخالفم': 3,
    'نظری ندارم': 4,
    'تا حدی موافقم': 5,
    'موافقم': 6,
    'کاملاً موافقم': 7,
    'قابل ارزیابی نیست / استفاده نکرده‌ام': np.nan,
    'قابل ارزیابی نیست': np.nan
}

# 3. تبدیل تمام مقادیر متنی به عددی در کل دیتافریم
df_numeric = df_clean.replace(likert_mapping)
df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

# 4. تعریف بلوک‌های پرسشنامه بر اساس شماره ستون‌های اکسل شما
# (حدوداً از ستون 30 به بعد گویه‌های لیکرت شروع می‌شوند. در صورت جابجایی ستون‌ها در اکسل جدید، اعداد زیر را تنظیم کنید)
constructs = {
    "شایستگی فنی (TC)": df_numeric.iloc[:, 30:35],
    "کیفیت تعامل (MM)": df_numeric.iloc[:, 35:39],
    "شفافیت (TR)": df_numeric.iloc[:, 39:43],
    "امنیت و کنترل (PC)": df_numeric.iloc[:, 43:47],
    "بومی‌سازی (CU)": df_numeric.iloc[:, 47:51],
    "اعتماد کلی (TRU)": df_numeric.iloc[:, 61:64]
}

print("=== گزارش پیش‌آزمون (Pilot Test) ===")
print(f"تعداد کل پاسخ‌دهندگان: {len(df_clean)} نفر\n")

# 5. محاسبه آلفای کرونباخ برای هر بلوک
for name, data in constructs.items():
    # حذف موقت رکوردهایی که در این بلوک مقدار خالی (مثل پردازش تصویر) دارند
    clean_data = data.dropna()
    
    if len(clean_data) > 0:
        # محاسبه آلفا با کتابخانه pingouin
        alpha = pg.cronbach_alpha(data=clean_data)[0]
        print(f"بلوک {name}:")
        print(f" - آلفای کرونباخ: {alpha:.3f}")
        if alpha >= 0.7:
            print(" - وضعیت: عالی (گویه‌ها کاملاً منسجم هستند)\n")
        else:
            print(" - وضعیت: نیازمند بررسی (احتمالاً یک سؤال برای کاربران گنگ بوده است)\n")
    else:
        print(f"بلوک {name}: دیتای کافی برای محاسبه وجود ندارد.\n")