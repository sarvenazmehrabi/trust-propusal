import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import glob
import os

# تنظیمات لاگر
logging.basicConfig(
    filename='../logs/eda_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# تعریف دیکشنری‌های توهم
HALLUCINATION_KEYWORDS = {
    'fa_ir': ['توهم', 'چرت', 'غلط', 'اشتباه', 'دروغ', 'ساختگی', 'نامربوط', 'الکی'],
    'en_us': ['hallucinat', 'fake', 'wrong', 'incorrect', 'made up', 'inaccurate', 'false']
}

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def flag_hallucination(text, lang):
    if not isinstance(text, str):
        return False
        
    text_lower = text.lower()
    keywords = HALLUCINATION_KEYWORDS.get(lang, [])
    
    for word in keywords:
        if word in text_lower:
            return True
    return False

def analyze_hallucination_impact(df, lang, chatbot_name):
    logging.info(f"Starting Hallucination EDA for {chatbot_name} ({lang})")
    
    df['has_hallucination_flag'] = df['content'].apply(lambda x: flag_hallucination(x, lang))
    
    total_comments = len(df)
    hallucinated_comments = df['has_hallucination_flag'].sum()
    hallucination_rate = (hallucinated_comments / total_comments) * 100 if total_comments > 0 else 0
    
    logging.info(f"{chatbot_name}: {hallucination_rate:.2f}% of comments mention hallucination/errors.")
    
    # مقایسه میانگین امتیاز با در نظر گرفتن احتمال خالی بودن دیتافریم
    normal_df = df[~df['has_hallucination_flag']]
    hallucinated_df = df[df['has_hallucination_flag']]
    
    avg_score_normal = normal_df['score'].mean() if not normal_df.empty else 0
    avg_score_hallucinated = hallucinated_df['score'].mean() if not hallucinated_df.empty else 0
    
    # مصورسازی
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='score', hue='has_hallucination_flag', palette='Set2')
    plt.title(f"Score Distribution by Hallucination Flag - {chatbot_name} ({lang})")
    plt.xlabel("User Score (Stars)")
    plt.ylabel("Number of Comments")
    plt.legend(title='Mentions Hallucination', labels=['No', 'Yes'])
    
    plt.savefig(f"../logs/{chatbot_name}_{lang}_hallucination_eda.png")
    plt.close()
    
    return {
        'chatbot': chatbot_name,
        'language': lang,
        'hallucination_rate_%': round(hallucination_rate, 2),
        'avg_score_normal': round(avg_score_normal, 2),
        'avg_score_hallucinated': round(avg_score_hallucinated, 2)
    }

if __name__ == "__main__":
    raw_data_dir = '../data/raw/'
    all_csv_files = glob.glob(os.path.join(raw_data_dir, '*.csv'))
    
    if not all_csv_files:
        print(f"خطا: هیچ فایل CSV در مسیر {raw_data_dir} پیدا نشد.")
    else:
        print(f"تعداد {len(all_csv_files)} فایل پیدا شد. شروع پردازش...")
        all_stats = []
        
        for filepath in all_csv_files:
            filename = os.path.basename(filepath)
            
            # تشخیص زبان از روی نام فایل
            if '_en_us' in filename:
                lang = 'en_us'
            else:
                lang = 'fa_ir' # فایل‌های کافه‌بازار و فایل‌های fa_ir
                
            # استخراج نام چت‌بات از روی الگوهای نام‌گذاری فایل‌ها
            parts = filename.replace('.csv', '').split('_')
            if 'balanced' in filename:
                chatbot_name = parts[2]
            elif 'native' in filename:
                chatbot_name = parts[2]
            else:
                chatbot_name = parts[0]
                
            print(f"در حال پردازش: {chatbot_name} ({lang})")
            
            df = load_data(filepath)
            stats = analyze_hallucination_impact(df, lang, chatbot_name)
            all_stats.append(stats)
            
        # نمایش نتایج در قالب یک جدول مرتب
        stats_df = pd.DataFrame(all_stats)
        print("\n=== خلاصه تحلیل توهم برای تمام چت‌بات‌ها ===")
        print(stats_df.to_string(index=False))