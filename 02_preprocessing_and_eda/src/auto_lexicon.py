import pandas as pd
import os
import glob
import re
from gensim.models import Word2Vec

def clean_text_and_remove_emojis(text):
    """
    حذف ایموجی‌ها، نمادها و نگه‌داشتن فقط حروف فارسی، انگلیسی و اعداد
    """
    text = str(text).lower()
    # این فرمول جادویی فقط حروف الفبای فارسی، انگلیسی و فاصله‌ها را نگه می‌دارد
    text = re.sub(r'[^\w\sآ-یa-z]', ' ', text)
    # حذف فاصله‌های اضافی
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_smart_lexicon(processed_dir, models_dir):
    print("۱. در حال بارگذاری داده‌ها و پاکسازی ایموجی‌ها...")
    base_files = [f for f in glob.glob(os.path.join(processed_dir, '*.csv')) 
                  if not f.endswith('_labeled.csv') and not f.endswith('_ready_for_ml.csv')]
    
    all_sentences = []
    
    for filepath in base_files:
        df = pd.read_csv(filepath)
        # پاکسازی محتوا
        df['ultra_clean_content'] = df['content'].apply(clean_text_and_remove_emojis)
        
        # استخراج جملات به صورت لیستِ کلمات برای تغذیه به شبکه عصبی
        sentences = [text.split() for text in df['ultra_clean_content'].tolist() if len(text.split()) > 2]
        all_sentences.extend(sentences)
        
    print(f"تعداد {len(all_sentences)} جمله معتبر برای آموزش شبکه عصبی استخراج شد.")
    
    print("۲. آموزش شبکه عصبی Word2Vec برای درک روابط کلمات (این مرحله چند ثانیه طول می‌کشد)...")
    model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=3, workers=4)
    
    # ذخیره مدل برای استفاده‌های بعدی
    model.save(os.path.join(models_dir, "word2vec_comments.model"))
    
    print("\n۳. در حال استخراج خودکار دیکشنری (Lexicon Expansion) ...\n")
    
    # کلمات بذر (Seed Words) برای هر شاخص
    aspect_seeds = {
        'چالش‌های زیرساختی و فیلترینگ': ['فیلتر', 'اینترنت', 'vpn', 'فیلترشکن'],
        'شایستگی چندرسانه‌ای': ['عکس', 'تصویر', 'image', 'picture', 'ویدیو'],
        'امنیت روانی و کنترل داده': ['امنیت', 'هک', 'حریم', 'اطلاعات', 'security'],
        'شایستگی فنی و عملیاتی': ['هنگ', 'باگ', 'ارور', 'سرعت', 'اشتباه', 'bug'],
        'بومی‌سازی و هنجارهای فرهنگی': ['فارسی', 'ایران', 'لحن', 'ادب', 'دوست'],
        'حساسیت دامنه و زمینه': ['پول', 'خرید', 'دانشگاه', 'دکتر', 'کد', 'بانک'],
        'مهندسی شفافیت و پارادوکس آن': ['منبع', 'دلیل', 'سورس', 'source'],
        'پویایی زمانی و حافظه بافتی': ['حافظه', 'یاد', 'فراموش', 'memory']
    }
    
    smart_dictionary = {}
    
    for aspect, seeds in aspect_seeds.items():
        expanded_words = set(seeds)
        for seed in seeds:
            try:
                # استخراج 10 کلمه هم‌معنی و پرتکرار برای هر کلمه بذر
                similar_words = model.wv.most_similar(seed, topn=10)
                for word, score in similar_words:
                    if score > 0.5:  # فقط کلماتی که شباهت بالای 50 درصد دارند
                        expanded_words.add(word)
            except KeyError:
                continue # اگر کلمه بذر در داده‌ها نبود، رد شو
                
        smart_dictionary[aspect] = list(expanded_words)
        
        print(f"--- {aspect} ---")
        print(f"کلمات یافت شده توسط هوش مصنوعی: {', '.join(smart_dictionary[aspect])}\n")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    build_smart_lexicon(processed_dir, models_dir)