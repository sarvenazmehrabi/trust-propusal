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
    """نگاشت دقیق کلمات فارسی به ۹ شاخص واقعی استخراج شده از مدل SHAP"""
    text = str(persian_text).replace('\u200c', ' ')
    if 'زیرساخت' in text or 'فیلتر' in text or 'شبکه' in text: return 'Infrastructure & Filtering'
    if 'فنی' in text or 'عملیات' in text or 'شایستگی' in text: return 'Technical Competence'
    if 'شفاف' in text or 'پارادوکس' in text or 'توضیح' in text: return 'Transparency Eng'
    if 'دامنه' in text or 'تخصص' in text or 'حساسیت' in text: return 'Domain Sensitivity'
    if 'چند' in text or 'رسانه' in text or 'محتوا' in text: return 'Multimedia Competence'
    if 'بومی' in text or 'فرهنگ' in text or 'محلی' in text: return 'Localization Culture'
    if 'روان' in text or 'امنیت' in text or 'حریم' in text: return 'Psychological Safety'
    if 'زمان' in text or 'حافظه' in text or 'استمرار' in text: return 'Temporal Memory'
    return text

def plot_shap_top4_features(filepath, ecosystem_name, output_dir):
    print(f"\n📊 در حال استخراج و رسم ۴ شاخص برترِ واقعی SHAP برای: {ecosystem_name}...")
    
    df = pd.read_csv(filepath)
    if not all(col in df.columns for col in ['date', 'Composite_Trust', 'Final_Aspect']):
        return
        
    df['Unified_Date'] = df['date'].apply(unify_dates)
    df = df.dropna(subset=['Unified_Date', 'Composite_Trust', 'Final_Aspect'])
    df = df[(df['Unified_Date'] >= '2025-10-01') & (df['Unified_Date'] <= '2026-06-30')]
    
    # ترجمه داده‌ها به شاخص‌های استاندارد مدل
    df['Aspect_EN'] = df['Final_Aspect'].apply(map_to_shap_features)
    
    # === فیلتر دقیق مبتنی بر ۴ شاخص برتر واقعی خروجی SHAP ===
    if 'Global' in ecosystem_name:
        top_4_aspects = ['Technical Competence', 'Multimedia Competence', 'Localization Culture', 'Transparency Eng']
    else:
        top_4_aspects = ['Infrastructure & Filtering', 'Domain Sensitivity', 'Multimedia Competence', 'Transparency Eng']
        
    df_top4 = df[df['Aspect_EN'].isin(top_4_aspects)].copy()
    
    if df_top4.empty: return
        
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
    save_path = os.path.join(output_dir, f'Feature_Breakdown_SHAP_TOP4_EXACT_{safe_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ نمودار نهایی با ۴ شاخص دقیق SHAP ذخیره شد: {save_path}")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    if os.path.exists(global_file): plot_shap_top4_features(global_file, 'Global AI Platforms', models_dir)
    if os.path.exists(native_file): plot_shap_top4_features(native_file, 'Native AI Platforms (Iran)', models_dir)