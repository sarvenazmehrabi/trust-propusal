import pandas as pd
import os
import glob
from gensim.models import Word2Vec

def discover_sentiment_words(processed_dir):
    print("۱. خواندن تمام داده‌های یکپارچه (واقعی و مصنوعی)...")
    final_files = glob.glob(os.path.join(processed_dir, 'ALL_*_BALANCED_ready_for_ml.csv'))
    
    all_sentences = []
    for filepath in final_files:
        df = pd.read_csv(filepath)
        # استخراج جملات به صورت لیست کلمات
        sentences = [str(text).split() for text in df['final_clean_content'].tolist()]
        all_sentences.extend(sentences)
        
    print(f"تعداد {len(all_sentences)} جمله برای آموزش شبکه عصبی آماده شد.")
    
    print("۲. در حال آموزش Word2Vec برای درک لحن و احساساتِ دامنه چت‌بات‌ها...")
    # پارامترها را کمی حساس‌تر کردیم تا کلمات دقیق‌تری پیدا کند
    model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=3, workers=4)
    
    print("\n۳. استخراج کلمات هم‌خانواده با بار احساسی مثبت و منفی:\n")
    
    # کلمات بذر (Seed) برای قطبیت‌های احساسی
    sentiment_seeds = {
        'Positive_Words (قطبیت مثبت)': ['عالی', 'خوب', 'دقیق', 'درست', 'سریع', 'راحت', 'قشنگ', 'good', 'great', 'awesome', 'accurate'],
        'Negative_Words (قطبیت منفی)': ['بد', 'کند', 'هنگ', 'باگ', 'اشتباه', 'دروغ', 'غلط', 'خراب', 'ضعیف', 'ارور', 'bad', 'slow', 'crash', 'error', 'wrong', 'fake']
    }
    
    for polarity, seeds in sentiment_seeds.items():
        print(f"--- {polarity} ---")
        expanded = set(seeds)
        for seed in seeds:
            try:
                # استخراج 8 کلمه مشابه برای هر کلمه بذر
                similar = model.wv.most_similar(seed, topn=8)
                for word, score in similar:
                    if score > 0.45:  # آستانه شباهت 45 درصد
                        expanded.add(word)
            except KeyError:
                continue
        
        print(f"کلمات یافت شده توسط هوش مصنوعی: {', '.join(expanded)}\n")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    discover_sentiment_words(processed_dir)