import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import jdatetime
import matplotlib.dates as mdates

def unify_dates(date_val):
    """تبدیل دقیق تاریخ‌های شمسی به میلادی"""
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
    except:
        return pd.to_datetime(date_val, errors='coerce')

def process_monthly_trust(filepath, platform_name):
    """پردازش و محاسبه میانگین اعتماد ماهانه"""
    df = pd.read_csv(filepath)
    if 'date' not in df.columns:
        return None
        
    df['Unified_Date'] = df['date'].apply(unify_dates)
    df = df.dropna(subset=['Unified_Date', 'Composite_Trust'])
    
    # فیلتر کردن داده‌های نامعتبر (فقط بازه معقول رساله را نگه می‌داریم)
    df = df[(df['Unified_Date'] >= '2025-10-01') & (df['Unified_Date'] <= '2026-06-30')]
    
    # تنظیم تاریخ به عنوان ایندکس برای نمونه‌برداری ماهانه
    df.set_index('Unified_Date', inplace=True)
    
    # محاسبه میانگین اعتماد در هر ماه (و تبدیل به درصد)
    monthly_trust = df['Composite_Trust'].resample('M').mean() * 100
    
    # تبدیل خروجی به دیتافریم مرتب
    result_df = monthly_trust.reset_index()
    result_df['Ecosystem'] = platform_name
    result_df.columns = ['Month', 'Trust_Level_Pct', 'Ecosystem']
    return result_df

def plot_comparative_monthly_trend(global_file, native_file, output_dir):
    print(f"\n{'='*60}")
    print("📈 در حال محاسبه روند ماهانه اعتماد (بدون نویز)...")
    
    df_global_monthly = process_monthly_trust(global_file, 'Global AI (ChatGPT, Grok)')
    df_native_monthly = process_monthly_trust(native_file, 'Native AI (Roboo, Zigap)')
    
    if df_global_monthly is None or df_native_monthly is None:
        print("❌ خطا در خواندن فایل‌ها.")
        return
        
    # ترکیب هر دو دیتافریم برای رسم در یک نمودار
    combined_df = pd.concat([df_global_monthly, df_native_monthly], ignore_index=True)
    combined_df = combined_df.dropna()
    
    # رسم نمودار خطیِ مقایسه‌ای
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fdfdfd"})
    
    ax = sns.lineplot(data=combined_df, x='Month', y='Trust_Level_Pct', hue='Ecosystem', 
                 palette=['#2c3e50', '#c0392b'], linewidth=3, marker='o', markersize=8)
    
    plt.title('Comparative Monthly Trend of User Trust in AI Platforms', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Timeline (Monthly Aggregation)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Trust Level (%)', fontsize=12, fontweight='bold')
    
    # تنظیمات فرمت تاریخ محور X (فقط نمایش نام ماه و سال)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
    plt.xticks(rotation=30)
    
    # تنظیم محور Y بین 0 تا 100 درصد
    plt.ylim(0, 100)
    
    plt.legend(title='Ecosystem', loc='center left', bbox_to_anchor=(1, 0.5), shadow=True)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'Monthly_Comparative_Trust_Trend.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ نمودار مقایسه‌ای ماهانه با موفقیت ذخیره شد: {save_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    if os.path.exists(global_file) and os.path.exists(native_file):
        plot_comparative_monthly_trend(global_file, native_file, models_dir)
    else:
        print("❌ فایل‌های داده یافت نشدند.")