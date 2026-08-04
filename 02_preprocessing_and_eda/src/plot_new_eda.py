import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_plot_data(df, platform_name):
    df_filtered = df[df['app_label'] == platform_name].copy()
    
    def categorize_trust(score):
        if score >= 4:
            return 'High_Trust (4-5 Stars)'
        elif score <= 2:
            return 'Low_Trust (1-2 Stars)'
        else:
            return 'Neutral'
            
    df_filtered['Trust_Level'] = df_filtered['score'].apply(categorize_trust)
    df_filtered = df_filtered[df_filtered['Trust_Level'] != 'Neutral']
    
    aspect_mapping = {
        'شایستگی فنی و عملیاتی': 'Technical Competence',
        'امنیت روانی و کنترل داده': 'Psychological Safety',
        'مهندسی شفافیت و پارادوکس آن': 'Transparency Eng',
        'بومی‌سازی و هنجارهای فرهنگی': 'Localization Culture',
        'پویایی زمانی و حافظه بافتی': 'Temporal Memory',
        'حساسیت دامنه و زمینه': 'Domain Sensitivity',
        'چالش‌های زیرساختی و فیلترینگ': 'Infrastructure & Filtering',
        'شایستگی چندرسانه‌ای': 'Multimedia Competence',
        'شاخص‌های رفتاری': 'Behavioral Indicators'
    }
    
    df_filtered['English_Aspect'] = df_filtered['Final_Aspect'].map(aspect_mapping).fillna(df_filtered['Final_Aspect'])
    
    total_comments = len(df_filtered)
    freq_df = df_filtered.groupby(['English_Aspect', 'Trust_Level']).size().reset_index(name='Count')
    freq_df['Percentage (%)'] = (freq_df['Count'] / total_comments) * 100
    
    return freq_df

def draw_and_save_plot(data, title, output_filename):
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    palette = {'High_Trust (4-5 Stars)': '#5cb85c', 'Low_Trust (1-2 Stars)': '#d9534f'}
    
    ax = sns.barplot(
        data=data, 
        y='English_Aspect', 
        x='Percentage (%)', 
        hue='Trust_Level',
        palette=palette,
        order=data['English_Aspect'].unique()
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Percentage of Comments Mentioning Category (%)', fontsize=12)
    plt.ylabel('Delphi & Emergent Categories', fontsize=12)
    
    plt.legend(title='User Satisfaction', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"نمودار با موفقیت ذخیره شد: {output_filename}")

def generate_eda_charts(processed_dir):
    print("شروع رسم نمودارهای جدید EDA...")
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    if os.path.exists(global_file):
        df_global = pd.read_csv(global_file)
        grok_data = prepare_plot_data(df_global, 'Grok')
        draw_and_save_plot(
            grok_data, 
            'Frequency of Categories in User Comments\nGrok (Global - Stable & High Engagement)',
            os.path.join(processed_dir, 'EDA_Fig_4_1_Grok.png')
        )
        
    if os.path.exists(native_file):
        df_native = pd.read_csv(native_file)
        roboo_data = prepare_plot_data(df_native, 'Roboo')
        draw_and_save_plot(
            roboo_data, 
            'Frequency of Categories in User Comments\nRoboo (Native - Critical)',
            os.path.join(processed_dir, 'EDA_Fig_4_2_Roboo_Final.png')
        )

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    generate_eda_charts(processed_dir)