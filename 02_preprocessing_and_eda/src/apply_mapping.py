import pandas as pd
import os
import glob

def process_and_map_labels(models_dir, processed_dir):
    print("\nشروع عملیات نگاشت و برچسب‌گذاری نهایی برای تمام قطعات داده (Train, Val, Test)...")
    
    # پیدا کردن تمام 54 فایل CSV که هنوز لیبل نهایی نخورده‌اند
    all_files = glob.glob(os.path.join(processed_dir, '*.csv'))
    target_files = [f for f in all_files if not f.endswith('_labeled.csv')]
    
    total_processed = 0
    for filepath in target_files:
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        
        # ساخت ستون جدید متغیر هدف
        df['Final_Aspect'] = 'Noise'
        
        # اعمال قوانین نگاشت قطعی استخراج شده از BERTopic
        for index, row in df.iterrows():
            text = str(row['cleaned_content']).lower()
            if text == 'nan' or not text.strip():
                continue
                
            best_match = 'Noise'
            
            if any(word in text for word in ['فیلتر', 'اینترنت', 'شکن', 'iran']):
                best_match = 'چالش‌های زیرساختی و فیلترینگ (اکتشافی)'
            elif any(word in text for word in ['image', 'photo', 'عکس', 'picture', 'video', 'generate']):
                best_match = 'شایستگی چندرسانه‌ای (اکتشافی)'
            elif any(word in text for word in ['هک', 'دزد', 'حریم', 'امنیت', 'hack', 'privacy', 'security']):
                best_match = 'امنیت روانی و کنترل داده'
            elif any(word in text for word in ['فحش', 'ادب', 'رباتی', 'فارسی', 'rude', 'polite', 'culture']):
                best_match = 'بومی‌سازی و هنجارهای فرهنگی'
            elif any(word in text for word in ['هنگ', 'باگ', 'سرعت', 'توهم', 'bug', 'crash', 'speed', 'hallucinate']):
                best_match = 'شایستگی فنی و عملیاتی'
            elif any(word in text for word in ['بانک', 'پول', 'دارو', 'دکتر', 'bank', 'money', 'doctor', 'code']):
                best_match = 'حساسیت دامنه و زمینه'
            elif any(word in text for word in ['دلیل', 'سورس', 'منبع', 'reason', 'source', 'reference']):
                best_match = 'مهندسی شفافیت و پارادوکس آن'
            elif any(word in text for word in ['یاد', 'فراموش', 'حافظه', 'remember', 'forget', 'memory']):
                best_match = 'پویایی زمانی و حافظه بافتی'
            else:
                if any(word in text for word in ['good', 'عالی', 'بد', 'bad', 'great', 'خوب', 'nice', 'awesome', 'خوبه']):
                    best_match = 'شاخص‌های رفتاری'
                    
            df.at[index, 'Final_Aspect'] = best_match
            
        total_processed += len(df)
        
        # ذخیره با استاندارد utf-8-sig برای نمایش صحیح فارسی در اکسل ویندوز
        output_path = os.path.join(processed_dir, filename.replace('.csv', '_labeled.csv'))
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"فایل {filename} با موفقیت برچسب‌گذاری شد.")

    print(f"\nعملیات با موفقیت پایان یافت. مجموع {total_processed} کامنت در تمام بخش‌ها برچسب‌گذاری شدند.")

if __name__ == "__main__":
    models_dir = os.path.join('..', 'models')
    processed_dir = os.path.join('..', 'data', 'processed')
    process_and_map_labels(models_dir, processed_dir)