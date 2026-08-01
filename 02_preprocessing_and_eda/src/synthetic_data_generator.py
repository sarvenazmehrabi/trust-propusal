import pandas as pd
import numpy as np
import os
import glob
import random
from datetime import datetime, timedelta
import uuid

def random_date(start_date, end_date):
    time_between_dates = end_date - start_date
    random_number_of_days = random.randrange(time_between_dates.days)
    gen_date = start_date + timedelta(
        days=random_number_of_days, 
        hours=random.randrange(24), 
        minutes=random.randrange(60),
        seconds=random.randrange(60)
    )
    return gen_date

def generate_username(lang):
    """تولید نام کاربری طبیعی به جای user_1234"""
    en_names = ['alex', 'sarah', 'mike', 'john', 'emma', 'chris', 'david', 'sam', 'jessica', 'tom', 'ali', 'reza', 'sara', 'maryam', 'amir', 'mehdi', 'zahra']
    separators = ['', '_', '.', '']
    if random.random() > 0.5:
        return f"{random.choice(en_names)}{random.choice(separators)}{random.randint(10, 9999)}"
    else:
        return f"{random.choice(en_names)}{random.choice(en_names)}{random.randint(1, 99)}"

def generate_synthetic_text(lang, aspect, polarity):
    """تولید متن با تنوع بالا (مثبت و منفی) و تمرکز روی دقت پاسخگویی"""
    if lang == 'fa':
        if aspect == 'شایستگی فنی و عملیاتی':
            if polarity == -1:
                subjects = ['برنامه', 'اپلیکیشن', 'ربات', 'هوش مصنوعی', 'نسخه جدید']
                verbs = ['هنگ میکنه', 'ارور میده', 'خیلی کنده', 'باگ داره', 'وسط کار میپره بیرون', 'اصلا لود نمیشه']
                adverbs = ['همیشه', 'مدام', 'بعضی وقتا', 'بدون دلیل', 'اکثرا']
                return f"{random.choice(subjects)} {random.choice(adverbs)} {random.choice(verbs)}"
            else:
                return random.choice(['سرعتش خیلی خوبه', 'روان و بدون باگ کار میکنه', 'عالی و سریع جواب میده', 'برنامه خوبیه هنگ نداره'])
                
        elif aspect == 'مهندسی شفافیت و پارادوکس آن':
            if polarity == -1:
                # تمرکز روی اشتباه بودن و دقت پایین طبق دستور شما
                subjects = ['جواباش', 'اطلاعاتش', 'راهنماییش', 'پاسخ ها', 'جوابایی که میده']
                verbs = ['اشتباهه', 'دقیق نیست', 'غلطه', 'پرت و پلا میگه', 'کاملا اشتباه راهنمایی میکنه', 'اصلا درست جواب نمیده']
                adverbs = ['خیلی وقتا', 'اکثرا', 'بیشتر مواقع', 'متاسفانه', 'تو بعضی زمینه ها']
                return f"{random.choice(adverbs)} {random.choice(subjects)} {random.choice(verbs)}"
            else:
                return random.choice(['جواباش کاملا درسته', 'خیلی دقیق راهنمایی میکنه', 'اطلاعاتش دقیقه', 'پاسخ دهی عالی و درست'])
                
        elif aspect == 'چالش‌های زیرساختی و فیلترینگ':
            if polarity == -1:
                return random.choice(['بدون فیلترشکن بالا نمیاد', 'همش قطعی داره', 'به اینترنت وصل نمیشه', 'سروراش قطعه'])
            else:
                return random.choice(['خوبه که بدون فیلترشکن کار میکنه', 'وصل شدنش راحته'])

        elif aspect == 'شایستگی چندرسانه‌ای':
            if polarity == -1:
                return random.choice(['عکس درست نمیسازه', 'تو ساخت تصویر ضعیفه', 'عکساش کیفیت نداره', 'اصلا نمیتونه عکس تولید کنه'])
            else:
                return random.choice(['عکسای خیلی قشنگی میسازه', 'کیفیت تصاویرش عالیه'])

    elif lang == 'en':
        if aspect == 'شایستگی فنی و عملیاتی':
            if polarity == -1:
                return random.choice(['app crashes a lot', 'too slow and buggy', 'gives errors randomly', 'keeps freezing'])
            else:
                return random.choice(['very fast and smooth', 'works perfectly without bugs', 'no crashes at all'])
                
        elif aspect == 'مهندسی شفافیت و پارادوکس آن':
            if polarity == -1:
                return random.choice(['answers are completely wrong', 'gives inaccurate information', 'fails to answer correctly', 'makes a lot of mistakes', 'not accurate at all'])
            else:
                return random.choice(['very accurate answers', 'gives precise information', 'correct and helpful responses'])
                
        elif aspect == 'شایستگی چندرسانه‌ای':
            if polarity == -1:
                return random.choice(['fails to generate images', 'bad at making pictures', 'image generation is awful'])
            else:
                return random.choice(['creates awesome images', 'good image generation', 'nice pictures'])
                
    return "good app" if polarity == 1 else "bad app"

def create_synthetic_dataframe(target_count, lang, is_native):
    rows = []
    
    if is_native:
        start_date = datetime(2025, 11, 22)
        end_date = datetime(2026, 5, 21)
        aspects = ['شایستگی فنی و عملیاتی', 'مهندسی شفافیت و پارادوکس آن', 'چالش‌های زیرساختی و فیلترینگ']
        app_labels = ['Atena', 'GapGPT', 'Hooshang', 'Roboo', 'Vira', 'Zigap']
    else:
        start_date = datetime(2025, 11, 1)
        end_date = datetime(2026, 4, 30)
        aspects = ['شایستگی فنی و عملیاتی', 'مهندسی شفافیت و پارادوکس آن', 'شایستگی چندرسانه‌ای']
        app_labels = ['ChatGPT', 'Gemini', 'Copilot', 'Grok', 'DeepSeek', 'Perplexity']
        
    for _ in range(target_count):
        gen_date = random_date(start_date, end_date)
        selected_aspect = random.choice(aspects)
        selected_app = random.choice(app_labels)
        
        # 70% کامنت‌ها انتقادی (-1) و 30% مثبت (1) هستند تا امتیازهای 4 و 5 هم داشته باشیم
        polarity = random.choices([-1, 1], weights=[0.7, 0.3])[0]
        gen_text = generate_synthetic_text(lang, selected_aspect, polarity)
        
        # تناسب قطبیت با ستاره‌ها
        if polarity == -1:
            score = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        else:
            score = random.choices([4, 5], weights=[0.4, 0.6])[0]
            
        row = {
            'review_id': str(uuid.uuid4())[:8] + '-syn',
            'user_name': generate_username(lang),
            'content': gen_text,
            'score': score,
            'full_date': gen_date.strftime('%Y-%m-%d %H:%M:%S'),
            'date': gen_date.strftime('%Y-%m-%d'),
            'time': gen_date.strftime('%H:%M:%S'),
            'app_version': f"1.{random.randint(0,9)}.{random.randint(10,999)}",
            'developer_reply': np.nan,  # ستون خالی برای پاسخ توسعه دهنده
            'app_label': selected_app,  # نام پلتفرم
            'lang': lang,
            'cleaned_content': gen_text,
            'final_clean_content': gen_text,
            'Final_Aspect': selected_aspect,
            'Sentiment_Polarity': polarity  # ستون صحیح احساسات به جای is_synthetic
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

def balance_datasets(processed_dir):
    print("شروع عملیات دیتاسازی بسیار طبیعی و هوشمند...")
    all_files = glob.glob(os.path.join(processed_dir, '*_ready_for_ml.csv'))
    final_files = [f for f in all_files if not os.path.basename(f).startswith('ALL_')]
    
    global_df_list = []
    native_df_list = []
    
    for f in final_files:
        df = pd.read_csv(f)
        # اگر در فایل‌های قبلی ستون Sentiment نبود، موقتا اضافه میکنیم که ارور ندهد
        if 'Sentiment_Polarity' not in df.columns:
            df['Sentiment_Polarity'] = 0 
            
        if 'native_data' in os.path.basename(f):
            native_df_list.append(df)
        else:
            global_df_list.append(df)
            
    global_df = pd.concat(global_df_list, ignore_index=True) if global_df_list else pd.DataFrame()
    native_df = pd.concat(native_df_list, ignore_index=True) if native_df_list else pd.DataFrame()
    
    # حذف ستون اضافه is_synthetic اگر از اجرای قبلی مانده باشد
    if 'is_synthetic' in global_df.columns: global_df.drop(columns=['is_synthetic'], inplace=True)
    if 'is_synthetic' in native_df.columns: native_df.drop(columns=['is_synthetic'], inplace=True)
    
    needed_global = max(0, 25721 - len(global_df))
    needed_native = max(0, 4240 - len(native_df))
    
    if needed_native > 0:
        synth_native_fa = create_synthetic_dataframe(needed_native, 'fa', is_native=True)
        native_df = pd.concat([native_df, synth_native_fa], ignore_index=True)
        
    if needed_global > 0:
        synth_global_en = create_synthetic_dataframe(needed_global // 2, 'en', is_native=False)
        synth_global_fa = create_synthetic_dataframe(needed_global // 2, 'fa', is_native=False)
        global_df = pd.concat([global_df, synth_global_en, synth_global_fa], ignore_index=True)
        
    global_output = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_output = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    global_df = global_df.sample(frac=1, random_state=42).reset_index(drop=True)
    native_df = native_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    global_df.to_csv(global_output, index=False, encoding='utf-8-sig')
    native_df.to_csv(native_output, index=False, encoding='utf-8-sig')
    
    print("عملیات با موفقیت پایان یافت! داده‌های مصنوعی ۱۰۰٪ مطابق با اسکیما و طبیعی تولید شدند.")
    print(f"تعداد نهایی جهانی: {len(global_df)} | بومی: {len(native_df)}")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    balance_datasets(processed_dir)