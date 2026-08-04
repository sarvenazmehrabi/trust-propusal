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

def map_exact_shap_features(persian_text):
    """تطبیق 100 درصدی و بدون خطای مقادیر دقیق دیتاست با شاخص‌های SHAP"""
    mapping = {
        'شایستگی فنی و عملیاتی': 'Technical Competence',
        'شایستگی چندرسانه‌ای': 'Multimedia Competence',
        'مهندسی شفافیت و پارادوکس آن': 'Transparency Eng',
        'چالش‌های زیرساختی و فیلترینگ': 'Infrastructure & Filtering',
        'حساسیت دامنه و زمینه': 'Domain Sensitivity',
        'بومی‌سازی و هنجارهای فرهنگی': 'Localization Culture',
        'پویایی زمانی و حافظه بافتی': 'Temporal Memory',
        'امنیت روانی و کنترل داده': 'Psychological Safety'
    }
    # حریم‌های خالیِ احتمالی را حذف می‌کند و مستقیماً ترجمه می‌کند
    return mapping.get(str(persian_text).strip(), str(persian_text))

def plot_shap_top4_flawless(filepath, ecosystem_name, output_dir):
    print(f"\n📊 در حال رسم دقیق ۴ شاخص برتر SHAP برای: {ecosystem_name}...")
    
    df = pd.read_csv(filepath)
    if not all(col in df.columns for col in ['date', 'Composite_Trust', 'Final_Aspect']):
        return
        
    df['Unified_Date'] = df['date'].apply(unify_dates)
    df = df.dropna(subset=['Unified_Date', 'Composite_Trust', 'Final_Aspect'])
    df = df[(df['Unified_Date'] >= '2025-10-01') & (df['Unified_Date'] <= '2026-06-30')]
    
    # اعمال ترجمه‌ی قطعی
    df['Aspect_EN'] = df['Final_Aspect'].apply(map_exact_shap_features)
    
    # هاردکد کردن دقیق ۴ شاخص برتر SHAP بر اساس خروجی مدل شما
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
    
    # رسم خطوط
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
    save_path = os.path.join(output_dir, f'Feature_Breakdown_SHAP_FLAWLESS_{safe_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ نمودار کامل با ۴ خط دقیق و بدون باگ ذخیره شد: {save_path}")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    print(f"{'='*60}")
    if os.path.exists(global_file): plot_shap_top4_flawless(global_file, 'Global AI Platforms', models_dir)
    if os.path.exists(native_file): plot_shap_top4_flawless(native_file, 'Native AI Platforms', models_dir)
    print(f"{'='*60}\n")