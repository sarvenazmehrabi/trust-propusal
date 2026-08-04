import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import jdatetime
import matplotlib.dates as mdates

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

def map_to_shap_features(persian_text):
    """نگاشت فوق‌هوشمند با پوشش تمام کلمات کلیدی احتمالی دیتاست"""
    text = str(persian_text).replace('\u200c', ' ')
    
    if any(w in text for w in ['زیرساخت', 'فیلتر', 'شبکه', 'اینترنت', 'قطعی', 'سرور', 'وصل']): 
        return 'Infrastructure & Filtering'
    if any(w in text for w in ['فنی', 'عملیات', 'شایستگی', 'سرعت', 'باگ', 'کد', 'برنامه']): 
        return 'Technical Competence'
    if any(w in text for w in ['شفاف', 'پارادوکس', 'توضیح', 'منبع', 'دلیل', 'چرا']): 
        return 'Transparency Eng'
    if any(w in text for w in ['دامنه', 'تخصص', 'حساسیت', 'علمی', 'دانش', 'دقیق']): 
        return 'Domain Sensitivity'
    # رفع باگ Multimedia: افزودن تمام کلمات مرتبط با مدیا
    if any(w in text for w in ['چند', 'رسانه', 'محتوا', 'صدا', 'تصویر', 'عکس', 'صوتی', 'فایل', 'ویدیو', 'تولید']): 
        return 'Multimedia Competence'
    if any(w in text for w in ['بومی', 'فرهنگ', 'محلی', 'زبان', 'فارسی', 'ایران']): 
        return 'Localization Culture'
    if any(w in text for w in ['روان', 'امنیت', 'حریم', 'خصوصی', 'همدلی', 'اخلاق', 'احساس']): 
        return 'Psychological Safety'
    if any(w in text for w in ['زمان', 'حافظه', 'استمرار', 'تاریخچه', 'یادآوری', 'گذشته']): 
        return 'Temporal Memory'
    
    return text # اگر ترجمه نشد، خود کلمه را برمی‌گرداند تا در دیباگ ببینیم

def plot_shap_top4_exact(filepath, ecosystem_name, output_dir):
    print(f"\n📊 در حال رسم دقیق ۴ شاخص برتر SHAP برای: {ecosystem_name}...")
    
    df = pd.read_csv(filepath)
    if not all(col in df.columns for col in ['date', 'Composite_Trust', 'Final_Aspect']):
        return
        
    df['Unified_Date'] = df['date'].apply(unify_dates)
    df = df.dropna(subset=['Unified_Date', 'Composite_Trust', 'Final_Aspect'])
    df = df[(df['Unified_Date'] >= '2025-10-01') & (df['Unified_Date'] <= '2026-06-30')]
    
    # اعمال ترجمه
    df['Aspect_EN'] = df['Final_Aspect'].apply(map_to_shap_features)
    
    # سیستم دیباگ: پیدا کردن کلماتی که ترجمه نشده‌اند
    unmapped = df[df['Aspect_EN'].str.contains(r'[آ-ی]')]['Final_Aspect'].unique()
    if len(unmapped) > 0:
        print(f"⚠️ هشدار: این شاخص‌ها در دیتاست شما ترجمه نشدند و جا ماندند: {unmapped}")
    
    # هاردکد کردن دقیق ۴ شاخص برتر از روی خروجی مدل شما
    if 'Global' in ecosystem_name:
        top_4_aspects = ['Technical Competence', 'Multimedia Competence', 'Localization Culture', 'Transparency Eng']
    else:
        top_4_aspects = ['Infrastructure & Filtering', 'Domain Sensitivity', 'Multimedia Competence', 'Transparency Eng']
        
    df_top4 = df[df['Aspect_EN'].isin(top_4_aspects)].copy()
    
    if df_top4.empty: 
        print("❌ خطای بحرانی: هیچ داده‌ای برای رسم یافت نشد!")
        return
        
    df_top4.set_index('Unified_Date', inplace=True)
    monthly_trust = df_top4.groupby(['Aspect_EN', pd.Grouper(freq='M')])['Composite_Trust'].mean() * 100
    monthly_trust = monthly_trust.reset_index()
    monthly_trust.columns = ['Top 4 SHAP Drivers', 'Month', 'Trust Score (%)']
    
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fdfdfd", "grid.linestyle": "--"})
    
    ax = sns.lineplot(data=monthly_trust, x='Month', y='Trust Score (%)', 
                      hue='Top 4 SHAP Drivers', palette='husl', 
                      linewidth=3, marker='o', markersize=9)
    
    plt.title(f'{ecosystem_name}: Longitudinal Impact of Top 4 SHAP Features on Trust', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Timeline (Monthly Aggregation)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Trust Score Driven by Feature (%)', fontsize=12, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
    plt.xticks(rotation=30)
    ax.set_ylim(0, 100)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', shadow=True, fontsize=11, title_fontsize=12)
    plt.tight_layout()
    
    safe_name = ecosystem_name.replace(' ', '_').replace('/', '_')
    save_path = os.path.join(output_dir, f'Feature_Breakdown_SHAP_ROOTCAUSED_{safe_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ نمودار کامل با ۴ خط دقیق ذخیره شد: {save_path}")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    print(f"{'='*60}")
    if os.path.exists(global_file): plot_shap_top4_exact(global_file, 'Global AI Platforms', models_dir)
    if os.path.exists(native_file): plot_shap_top4_exact(native_file, 'Native AI Platforms', models_dir)
    print(f"{'='*60}\n")