import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# تنظیمات لاگر استاندارد دکتری
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
    """بارگذاری داده‌ها و مدیریت خطاهای احتمالی"""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def flag_hallucination(text, lang):
    """
    بررسی وجود کلمات مرتبط با توهم در متن کامنت.
    ورودی متنی است و خروجی یک مقدار بولین (True/False).
    """
    if not isinstance(text, str):
        return False
        
    text_lower = text.lower()
    keywords = HALLUCINATION_KEYWORDS.get(lang, [])
    
    for word in keywords:
        if word in text_lower:
            return True
    return False

def analyze_hallucination_impact(df, lang, chatbot_name):
    """
    تحلیل تاثیر توهم بر امتیاز (Score) کاربران و مصورسازی آن.
    """
    logging.info(f"Starting Hallucination EDA for {chatbot_name} ({lang})")
    
    # اعمال پرچم‌گذاری روی کامنت‌ها
    df['has_hallucination_flag'] = df['content'].apply(lambda x: flag_hallucination(x, lang))
    
    # محاسبه آمار
    total_comments = len(df)
    hallucinated_comments = df['has_hallucination_flag'].sum()
    hallucination_rate = (hallucinated_comments / total_comments) * 100
    
    logging.info(f"{chatbot_name}: {hallucination_rate:.2f}% of comments mention hallucination/errors.")
    
    # مقایسه میانگین امتیاز
    avg_score_normal = df[~df['has_hallucination_flag']]['score'].mean()
    avg_score_hallucinated = df[df['has_hallucination_flag']]['score'].mean()
    
    # مصورسازی
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='score', hue='has_hallucination_flag', palette='Set2')
    plt.title(f"Score Distribution by Hallucination Flag - {chatbot_name}")
    plt.xlabel("User Score (Stars)")
    plt.ylabel("Number of Comments")
    plt.legend(title='Mentions Hallucination', labels=['No', 'Yes'])
    
    # ذخیره نمودار
    plt.savefig(f"../logs/{chatbot_name}_{lang}_hallucination_eda.png")
    plt.close()
    
    return {
        'chatbot': chatbot_name,
        'hallucination_rate_%': round(hallucination_rate, 2),
        'avg_score_normal': round(avg_score_normal, 2),
        'avg_score_hallucinated': round(avg_score_hallucinated, 2)
    }

# نمونه اجرای اسکریپت (می‌توانید برای تمام فایل‌ها حلقه بنویسید)
if __name__ == "__main__":
    # مثال برای داده‌های Chatgpt فارسی
    df_chatgpt_fa = load_data('../data/raw/balanced_data_ChatGPT_fa_ir.csv')
    stats = analyze_hallucination_impact(df_chatgpt_fa, 'fa_ir', 'ChatGPT')
    print(stats)