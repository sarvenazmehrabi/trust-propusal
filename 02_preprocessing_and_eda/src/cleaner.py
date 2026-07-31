import pandas as pd
import os
import glob
import re
import logging
from sklearn.model_selection import StratifiedShuffleSplit

# کتابخانه‌های NLP
import hazm
import spacy

# تنظیمات لاگر
logging.basicConfig(
    filename=os.path.join('..', 'logs', 'preprocessing_pipeline.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# بارگذاری مدل‌های زبانی
print("در حال بارگذاری مدل‌های زبانی...")
nlp_en = spacy.load("en_core_web_sm")
normalizer_fa = hazm.Normalizer()
lemmatizer_fa = hazm.Lemmatizer()

# لیست سفارشی کلمات توقف (حفظ کلمات نافی)
STOPWORDS_FA = set(hazm.stopwords_list()) - {'نه', 'هیچ', 'نمی', 'نیست', 'ندارد', 'بدون', 'مگر'}
for word in ['not', 'no', 'never', 'none', 'neither', 'nor', 'without']:
    nlp_en.vocab[word].is_stop = False

def clean_persian_text(text):
    """خط لوله پیش‌پردازش استاندارد متون فارسی"""
    if pd.isna(text): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\.\!\؟\?]', '', text)
    text = normalizer_fa.normalize(text)
    tokens = hazm.word_tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        if token not in STOPWORDS_FA and len(token) > 1:
            lemma = lemmatizer_fa.lemmatize(token).split('#')[0] 
            cleaned_tokens.append(lemma)
    return " ".join(cleaned_tokens)

def clean_english_text(text):
    """خط لوله پیش‌پردازش استاندارد متون انگلیسی"""
    if pd.isna(text): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    doc = nlp_en(text.lower())
    cleaned_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]
    return " ".join(cleaned_tokens)

def stratified_split_and_save(df, filename, output_dir):
    """تقسیم‌بندی داده‌ها با مدیریت قطعی کلاس‌های اقلیت (آستانه امن ۱۰ عضو)"""
    df = df[df['cleaned_content'].str.strip().astype(bool)].copy()
    df = df.dropna(subset=['score'])
    
    # تبدیل اجباری امتیازات به عدد صحیح
    df['score'] = df['score'].astype(int)
    
    if len(df) < 10:
        logging.warning(f"داده‌های {filename} برای قطعه‌بندی بسیار کم است.")
        return
        
    # ارتقای آستانه به 10 برای عبور امن از تمام تقسیمات 15 درصدی
    class_counts = df['score'].value_counts()
    sparse_classes = class_counts[class_counts < 10].index.tolist()
    
    if sparse_classes:
        majority_class = class_counts.idxmax()
        logging.info(f"رفع قطبی‌شدگی در {filename}: ادغام کلاس‌های اقلیت {sparse_classes} با کلاس اکثریت ({majority_class})")
        df['score'] = df['score'].apply(lambda x: majority_class if x in sparse_classes else x)
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    
    try:
        for train_index, temp_index in splitter.split(df, df['score']):
            train_df = df.iloc[train_index]
            temp_df = df.iloc[temp_index]
            
        val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        for val_index, test_index in val_splitter.split(temp_df, temp_df['score']):
            val_df = temp_df.iloc[val_index]
            test_df = temp_df.iloc[test_index]
            
        base_name = filename.replace('.csv', '')
        train_df.to_csv(os.path.join(output_dir, f"{base_name}_train.csv"), index=False, encoding='utf-8')
        val_df.to_csv(os.path.join(output_dir, f"{base_name}_val.csv"), index=False, encoding='utf-8')
        test_df.to_csv(os.path.join(output_dir, f"{base_name}_test.csv"), index=False, encoding='utf-8')
        
        logging.info(f"{base_name} Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except ValueError as e:
        logging.error(f"خطا در قطعه‌بندی {filename}: {e}")

if __name__ == "__main__":
    raw_dir = os.path.join('..', 'data', 'raw')
    processed_dir = os.path.join('..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    all_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    print(f"شروع پیش‌پردازش و قطعه‌بندی {len(all_files)} فایل...")
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        print(f"در حال پردازش: {filename}")
        df = pd.read_csv(filepath)
        if '_en_us' in filename:
            df['cleaned_content'] = df['content'].apply(clean_english_text)
        else:
            df['cleaned_content'] = df['content'].apply(clean_persian_text)
        stratified_split_and_save(df, filename, processed_dir)
        
    print("عملیات پیش‌پردازش با موفقیت به پایان رسید. لاگ‌ها را بررسی کنید.")