import pandas as pd
import os
import glob

def is_generic_or_short(text):
    """
    بررسی می‌کند که آیا کامنت تک‌کلمه‌ای، کوتاه یا صرفاً شامل کلمات زرد است یا خیر.
    """
    if pd.isna(text):
        return True
    
    text = str(text).strip().lower()
    words = text.split()
    
    # قانون 1: اگر نظر فقط 1 یا 2 کلمه است، معمولا فاقد استدلال است (مثل: "خیلی عالی"، "افتضاح بود")
    if len(words) <= 2:
        return True
        
    # قانون 2: لیست کلمات زرد و فاقد ارزش تحلیلی برای رسنجش اعتماد
    generic_words = {
        'عالی', 'خوب', 'بد', 'افتضاح', 'مزخرف', 'چرت', 'خوبه', 'عالیه', 'بدک', 'مرسی', 'ممنون', 'تشکر',
        'good', 'bad', 'nice', 'great', 'awesome', 'terrible', 'ok', 'okay', 'perfect', 'super', 'thanks'
    }
    
    # اگر کل کلمات جمله فقط از این کلمات زرد و حروف ربط ساده تشکیل شده باشد
    words_set = set(words)
    if words_set.issubset(generic_words):
        return True
        
    return False

def filter_dataset(processed_dir):
    print("شروع فیلترینگ هوشمند: حذف کامنت‌های زرد، کوتاه و فاقد استدلال...")
    
    # فایل های پایه را می خوانیم (فایل هایی که هنوز لیبل نهایی ماشین را نخورده اند)
    # یعنی فایل هایی که با _labeled یا _ready_for_ml تمام نمی شوند
    base_files = [f for f in glob.glob(os.path.join(processed_dir, '*.csv')) 
                  if not f.endswith('_labeled.csv') and not f.endswith('_ready_for_ml.csv')]
    
    total_before = 0
    total_after = 0
    
    for filepath in base_files:
        df = pd.read_csv(filepath)
        count_before = len(df)
        total_before += count_before
        
        # اعمال فیلتر روی ستون محتوای اصلی (content)
        # علامت ~ یعنی آنهایی که generic نیستند را نگه دار
        mask = ~df['content'].apply(is_generic_or_short)
        filtered_df = df[mask].copy()
        
        count_after = len(filtered_df)
        total_after += count_after
        
        # ذخیره مجدد روی همان فایل پایه با فرمت استاندارد
        filtered_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
    deleted_count = total_before - total_after
    
    print("\n==================================================")
    print(f"گزارش عملیات پاکسازی عمیق (Deep Filtering):")
    print(f"تعداد کل کامنت‌ها قبل از فیلتر: {total_before}")
    print(f"تعداد کامنت‌های حذف شده (زرد و کوتاه): {deleted_count} ({(deleted_count/total_before)*100:.1f}%)")
    print(f"تعداد کامنت‌های استدلال‌دار و خالص باقیمانده: {total_after}")
    print("==================================================")
    print("\nحالا داده‌های شما طلای خالص هستند!")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    filter_dataset(processed_dir)