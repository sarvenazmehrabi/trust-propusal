import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def clean_and_extract_major_minor(version_str):
    """استخراج نسخه اصلی و فرعی (Major.Minor) برای یکپارچگی لیبل‌ها"""
    try:
        version_str = str(version_str).strip()
        if pd.isna(version_str) or 'unknown' in version_str.lower() or 'null' in version_str.lower():
            return None
        # استخراج دو عدد اول جدا شده با نقطه (مثل 1.5 یا 30.0)
        match = re.search(r'^(\d+\.\d+)', version_str)
        if match:
            return float(match.group(1)) # تبدیل به اعشاری برای مرتب‌سازی ریاضی دقیق
        return None
    except:
        return None

def analyze_app_versions_cleaned(filepath):
    df = pd.read_csv(filepath)
    if 'app_version' not in df.columns or 'Composite_Trust' not in df.columns:
        return None
        
    df = df.dropna(subset=['app_version', 'Composite_Trust'])
    
    # اعمال تابع پاکسازی و استخراج نسخه
    df['Major_Minor_Version'] = df['app_version'].apply(clean_and_extract_major_minor)
    df = df.dropna(subset=['Major_Minor_Version'])
    
    # محاسبه میانگین اعتماد برای هر گروه از نسخه‌های اصلی
    trust_by_version = df.groupby('Major_Minor_Version')['Composite_Trust'].mean() * 100
    
    # مرتب‌سازیِ کاملاً ریاضی و صعودی بر اساس شماره نسخه (نه بر اساس تاریخ کامنت)
    ordered_trust = trust_by_version.sort_index().reset_index()
    ordered_trust.columns = ['App_Version', 'Trust_Level_Pct']
    
    # فیلتر کردن: فقط نسخه‌هایی که روند معناداری دارند را نگه می‌داریم (حذف نویزهای ابتدایی/انتهایی)
    # برای حفظ زیبایی، 15 نسخه متوالی و پرتکرار را می‌گیریم
    if len(ordered_trust) > 15:
        # انتخاب 15 نسخه با بیشترین تراکم کاربر
        top_versions = df['Major_Minor_Version'].value_counts().nlargest(15).index
        ordered_trust = ordered_trust[ordered_trust['App_Version'].isin(top_versions)].sort_values('App_Version')
        
    # تبدیل مجدد به رشته برای نمایش در نمودار
    ordered_trust['App_Version'] = ordered_trust['App_Version'].astype(str)
    return ordered_trust

def plot_enhanced_journal_version(global_file, native_file, output_dir):
    print(f"\n{'='*60}")
    print("✨ در حال پردازش نسخه‌های Major.Minor و رسم نمودار با Annotations...")
    
    global_v_data = analyze_app_versions_cleaned(global_file)
    native_v_data = analyze_app_versions_cleaned(native_file)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharey=True)
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fbfcfc", "grid.linestyle": "--", "grid.alpha": 0.6})
    
    # === نمودار اول: جهانی ===
    if global_v_data is not None and not global_v_data.empty:
        sns.pointplot(ax=axes[0], data=global_v_data, x='App_Version', y='Trust_Level_Pct', 
                      color='#1f4e79', scale=1.3, linestyles='-', marker='o')
        
        axes[0].set_title('A. Global AI Platforms: Trust Fluctuation Across Major Software Updates', 
                          fontsize=15, fontweight='bold', pad=15)
        axes[0].set_ylabel('Average Trust Level (%)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('')
        axes[0].tick_params(axis='x', rotation=30, labelsize=11)
        axes[0].fill_between(range(len(global_v_data)), global_v_data['Trust_Level_Pct'], color='#1f4e79', alpha=0.08)

    # === نمودار دوم: بومی ===
    if native_v_data is not None and not native_v_data.empty:
        sns.pointplot(ax=axes[1], data=native_v_data, x='App_Version', y='Trust_Level_Pct', 
                      color='#a93226', scale=1.3, linestyles='-', marker='s')
        
        axes[1].set_title('B. Native AI Platforms (Iran): Trust Fluctuation Across Major Software Updates', 
                          fontsize=15, fontweight='bold', pad=15)
        axes[1].set_ylabel('Average Trust Level (%)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Major App Release Version (Sequential Order)', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=30, labelsize=11)
        axes[1].fill_between(range(len(native_v_data)), native_v_data['Trust_Level_Pct'], color='#a93226', alpha=0.08)
        
        # --- سیستم حاشیه‌نویسی هوشمند (Event Annotations) ---
        # پیدا کردن نقاطی که اعتماد در آن‌ها کمتر از 10 درصد شده است (Critical Crashes)
        critical_crashes = native_v_data[native_v_data['Trust_Level_Pct'] < 10]
        
        for idx, row in critical_crashes.iterrows():
            # پیدا کردن ایندکس واقعی در محور X
            x_pos = native_v_data.index.get_loc(idx)
            axes[1].annotate('System/Filtering Shock', 
                             xy=(x_pos, row['Trust_Level_Pct']),
                             xytext=(x_pos, row['Trust_Level_Pct'] + 25),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                             fontsize=10, fontweight='bold', color='#a93226',
                             horizontalalignment='center',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#a93226", alpha=0.8))

    plt.ylim(0, 100)
    plt.tight_layout(pad=3.0)
    
    save_path = os.path.join(output_dir, 'Enhanced_Journal_App_Version_Impact.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ نمودار ارتقایافته با موفقیت ذخیره شد: {save_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    if os.path.exists(global_file) and os.path.exists(native_file):
        plot_enhanced_journal_version(global_file, native_file, models_dir)