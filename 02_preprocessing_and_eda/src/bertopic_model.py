import pandas as pd
import numpy as np
import os
import glob
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

def load_all_training_data(processed_dir):
    """بارگذاری تمامی داده‌های آموزش"""
    print("در حال بارگذاری داده‌های آموزش...")
    train_files = glob.glob(os.path.join(processed_dir, '*_train.csv'))
    docs = []
    for filepath in train_files:
        df = pd.read_csv(filepath)
        valid_docs = df['cleaned_content'].dropna().astype(str).tolist()
        docs.extend(valid_docs)
    return docs

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    docs = load_all_training_data(processed_dir)
    
    if len(docs) > 0:
        seed_topic_list = [
            # 0: امنیت روانی
            ["هک", "دزد", "حریم", "جاسوس", "لو", "شنود", "خطر", "ترس", "فیک", "کلاهبردار", "امنیت", "سیو", "اطلاعات", "شخصی", "پاک", "رصد", "ویروس", "بدافزار", "کپی", "دسترسی", "hack", "thief", "privacy", "spy", "leak", "eavesdrop", "danger", "fear", "fake", "scammer", "security", "save", "info", "personal", "delete", "track", "virus", "malware", "copy", "access"],
            # 1: بومی‌سازی
            ["صمیمی", "دوستانه", "توهین", "ادب", "فحش", "لحن", "چرت", "فارسی", "دری‌وری", "باحال", "بی‌ادب", "رباتی", "خشک", "بامزه", "کتابی", "عامیانه", "گیج", "خنگ", "شعور", "درک", "intimate", "friendly", "insult", "polite", "swear", "tone", "nonsense", "Persian", "gibberish", "cool", "rude", "robotic", "rigid", "funny", "formal", "slang", "confused", "dumb", "intellect", "understanding"],
            # 2: پویایی زمانی
            ["یاد", "فراموش", "سابقه", "حفظ", "آلزایمر", "حافظه", "قبل", "تاریخچه", "تکرار", "دوباره", "خالی", "یادآوری", "گمشده", "پرید", "برگشت", "قدیم", "جدید", "آپدیت", "دیروز", "الان", "remember", "forget", "history", "keep", "Alzheimer", "memory", "before", "record", "repeat", "again", "empty", "reminder", "lost", "wiped", "return", "old", "new", "update", "yesterday", "now"],
            # 3: حساسیت دامنه
            ["پول", "بانک", "دکتر", "بورس", "نسخه", "دارو", "وام", "دلار", "طلا", "مریض", "وکیل", "قانونی", "حرام", "حلال", "مالی", "ارز", "ترید", "کنکور", "مقاله", "کد", "money", "bank", "doctor", "stock", "prescription", "drug", "loan", "dollar", "gold", "patient", "lawyer", "legal", "forbidden", "halal", "financial", "crypto", "trade", "exam", "paper", "code"],
            # 4: شاخص‌های رفتاری
            ["مطمئن", "تکیه", "همیشه", "دروغ", "راست", "اعتماد", "کمک", "جواب", "چاخان", "الکی", "خالی‌بند", "خفن", "خوب", "مسخره", "سرکاری", "مفید", "لازم", "ضروری", "رفیق", "پارتنر", "sure", "rely", "always", "lie", "true", "trust", "help", "answer", "bullshit", "fake", "bullshitter", "awesome", "good", "ridiculous", "prank", "useful", "necessary", "essential", "buddy", "partner"],
            # 5: شایستگی فنی
            ["دقت", "سرعت", "هنگ", "توهم", "باگ", "کرش", "لگ", "کند", "سریع", "قطعی", "وصل", "لود", "نت", "ارور", "خطا", "نصب", "اجرا", "پولی", "رایگان", "ضعیف", "accuracy", "speed", "freeze", "hallucination", "bug", "crash", "lag", "slow", "fast", "disconnect", "connect", "load", "internet", "error", "mistake", "install", "run", "paid", "free", "weak"],
            # 6: مهندسی شفافیت
            ["دلیل", "سورس", "پنهان", "منبع", "چرا", "چطور", "توضیح", "لینک", "سایت", "نامعلوم", "مبهم", "روشن", "واضح", "سانسور", "فیلتر", "محدود", "قفل", "باز", "بسته", "اثبات", "reason", "source", "hidden", "reference", "why", "how", "explain", "link", "site", "unknown", "vague", "clear", "obvious", "censor", "filter", "limit", "lock", "open", "closed", "proof"]
        ]

        # 1. ساخت برچسب‌های نیمه‌نظارتی (ابتکار علمی جدید)
        print("\n>>> در حال اسکن کامنت‌ها و ساخت لنگرهای نیمه‌نظارتی (Semi-Supervised Labels)...")
        y = np.full(len(docs), -1)
        for i, doc in enumerate(docs):
            doc_words = set(doc.split())
            best_cat = -1
            max_hits = 0
            for cat_idx, words in enumerate(seed_topic_list):
                hits = len(doc_words.intersection(set(words)))
                if hits > max_hits:
                    max_hits = hits
                    best_cat = cat_idx
            y[i] = best_cat
        
        labeled_count = np.sum(y != -1)
        print(f"فاز کیفی با موفقیت به فاز کمی متصل شد.")
        print(f"تعداد کامنت‌های لنگرگذاری شده به عنوان تقلب ماشین: {labeled_count} از {len(docs)}")

        # 2. بارگذاری بردارها از هارد به جای محاسبه مجدد
        embeddings_path = os.path.join(models_dir, 'embeddings.npy')
        if os.path.exists(embeddings_path):
            print("\n>>> بارگذاری بردارها از هارد (صرفه‌جویی در زمان)...")
            embeddings = np.load(embeddings_path)
        else:
            print("خطا: فایل بردارها یافت نشد!")
            exit()
        
        # 3. مدل‌سازی هوشمند بدون باگِ ریاضی
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=300, min_samples=20, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        print("\nشروع آموزش مدل BERTopic (استراتژی Semi-Supervised)...")
        topic_model = BERTopic(
            language="multilingual",
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True
        )
        
        # تزریق برچسب‌ها (y) برای هدایت الگوریتم
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings, y=y)
        
        print("\nدر حال ذخیره‌سازی اطلاعات خوشه‌ها...")
        topic_info = topic_model.get_topic_info()
        
        print("\n=== نتایج خوشه‌بندی (10 خوشه برتر) ===")
        print(topic_info.head(10))
        
        topic_info.to_csv(os.path.join(models_dir, 'bertopic_clusters_info.csv'), index=False, encoding='utf-8')
        print(f"\nفایل نهایی در مسیر {models_dir}/bertopic_clusters_info.csv ذخیره شد.")