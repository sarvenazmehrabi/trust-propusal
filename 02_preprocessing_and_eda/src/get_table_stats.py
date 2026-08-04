import pandas as pd
import os
import glob
import numpy as np

def generate_table_stats(processed_dir):
    print("در حال استخراج آمار دقیق برای جدول ۳-۱ رساله...")
    final_files = glob.glob(os.path.join(processed_dir, 'ALL_*_BALANCED_ready_for_ml.csv'))
    
    all_data = []
    for f in final_files:
        df = pd.read_csv(f)
        ecosystem = 'Native' if 'NATIVE' in f else 'Global'
        df['Ecosystem'] = ecosystem
        all_data.append(df)
        
    if not all_data:
        print("فایل‌های نهایی یافت نشدند!")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # محاسبه تعداد کلمات هر کامنت
    full_df['Word_Count'] = full_df['final_clean_content'].astype(str).apply(lambda x: len(x.split()))
    
    # گروه بندی بر اساس اکوسیستم و نام چت بات
    stats = full_df.groupby(['Ecosystem', 'app_label']).agg(
        Total_Reviews=('review_id', 'count'),
        Avg_Words_per_Comment=('Word_Count', lambda x: round(x.mean(), 1)),
        Mean_Score=('score', lambda x: round(x.mean(), 2)),
        Score_Std_Dev=('score', lambda x: round(x.std(), 2)),
        Start_Date=('date', 'min'),
        End_Date=('date', 'max')
    ).reset_index()
    
    # مرتب سازی و نمایش زیباتر
    stats = stats.sort_values(by=['Ecosystem', 'Total_Reviews'], ascending=[True, False])
    
    print("\n--- آمار استخراج شده جهت درج در جدول Word ---")
    print(stats.to_string(index=False))
    
    # ذخیره در یک فایل اکسل برای راحتی کار شما
    output_path = os.path.join(processed_dir, 'Table_3_1_Stats.csv')
    stats.to_csv(output_path, index=False)
    print(f"\nآمار در فایل {output_path} ذخیره شد. میتوانید مستقیما در Word کپی کنید.")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    generate_table_stats(processed_dir)