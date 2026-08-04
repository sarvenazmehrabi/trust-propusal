import pandas as pd
import os
import jdatetime

def unify_dates(date_val):
    if pd.isna(date_val): return pd.NaT
    d_str = str(date_val).split(' ')[0].replace('/', '-')
    parts = d_str.split('-')
    try:
        if len(parts[0]) == 4 and 1300 <= int(parts[0]) <= 1500:
            jd = jdatetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
            g_date = jd.togregorian()
            return pd.to_datetime(f"{g_date.year}-{g_date.month}-{g_date.day}")
        elif len(parts[-1]) == 4 and 1300 <= int(parts[-1]) <= 1500:
            jd = jdatetime.date(int(parts[-1]), int(parts[1]), int(parts[0]))
            g_date = jd.togregorian()
            return pd.to_datetime(f"{g_date.year}-{g_date.month}-{g_date.day}")
        else:
            return pd.to_datetime(date_val, errors='coerce')
    except: return pd.to_datetime(date_val, errors='coerce')

def run_diagnostic(filepath, ecosystem_name):
    print(f"\n{'='*60}")
    print(f"🔍 شروع دیباگ و کالبدشکافی برای: {ecosystem_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ فایل پیدا نشد: {filepath}")
        return
        
    df = pd.read_csv(filepath)
    print(f"✅ تعداد کل ردیف‌های اولیه: {len(df)}")
    
    df['Unified_Date'] = df['date'].apply(unify_dates)
    df = df.dropna(subset=['Unified_Date', 'Composite_Trust', 'Final_Aspect'])
    print(f"✅ تعداد ردیف‌ها بعد از حذف مقادیر خالی: {len(df)}")
    
    df = df[(df['Unified_Date'] >= '2025-10-01') & (df['Unified_Date'] <= '2026-06-30')]
    print(f"✅ تعداد ردیف‌ها در بازه زمانی نمودار (اکتبر 2025 تا ژوئن 2026): {len(df)}")
    
    print("\n📊 لیست دقیقِ مقادیر خام ستون [Final_Aspect] در این بازه:")
    raw_aspects = df['Final_Aspect'].value_counts()
    
    for aspect, count in raw_aspects.items():
        print(f"   - '{aspect}': {count} ردیف")
        
if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    run_diagnostic(global_file, 'Global AI Platforms')
    run_diagnostic(native_file, 'Native AI Platforms')
    print(f"\n{'='*60}")