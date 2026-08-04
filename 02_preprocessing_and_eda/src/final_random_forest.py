import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def prepare_data(df):
    """
    مهندسی ویژگی‌ها: ترکیب جنبه‌ها با بار احساسی ($X$)
    و آماده‌سازی امتیاز کاربران به عنوان متغیر هدف ($Y$)
    """
    # حذف رکوردهای ناقص احتمالی
    df = df.dropna(subset=['score', 'Sentiment_Polarity', 'Final_Aspect']).copy()
    
    # تبدیل شاخص‌ها به متغیرهای دامی (One-Hot Encoding)
    features = pd.get_dummies(df['Final_Aspect'], dtype=float)
    
    # ضربِ برچسب شاخص در قطبیت احساسی
    # با این کار، انتقاد از سرعت مقدار 1- و رضایت از سرعت مقدار 1 می‌گیرد
    for col in features.columns:
        features[col] = features[col] * df['Sentiment_Polarity']
        
    X = features
    y = df['score'].astype(int)
    
    return X, y

def train_and_extract_insights(filepath, model_title, output_dir):
    print(f"\n==================================================")
    print(f">>> در حال آموزش مدل برای: {model_title}")
    
    df = pd.read_csv(filepath)
    X, y = prepare_data(df)
    
    print(f"تعداد کل رکوردهای معتبر تغذیه‌شده به مدل: {len(X)}")
    
    # تقسیم داده‌ها به آموزش و آزمون
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تعریف مدل جنگل تصادفی (تنظیمات بهینه‌شده برای جلوگیری از بیش‌برازش)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # سنجش دقت
    accuracy = rf.score(X_test, y_test)
    print(f"دقت پیش‌بینی مدل (Accuracy): {accuracy:.2f}")
    
    # استخراج اهمیت ویژگی‌ها (Feature Importance)
    importances = rf.feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Aspect': X.columns,
        'Importance_Weight': importances
    }).sort_values(by='Importance_Weight', ascending=False)
    
    print(f"\n--- رتبه‌بندی شاخص‌های اعتماد در {model_title} ---")
    
    # محاسبه درصد اهمیت برای نمایش زیباتر
    feat_imp_df['Importance_Percentage'] = (feat_imp_df['Importance_Weight'] * 100).round(2).astype(str) + ' %'
    print(feat_imp_df[['Aspect', 'Importance_Percentage']].to_string(index=False))
    
    # ذخیره خروجی‌ها در پوشه مدل‌ها
    safe_title = model_title.replace(' ', '_')
    output_path = os.path.join(output_dir, f'{safe_title}_feature_importance.csv')
    feat_imp_df.to_csv(output_path, index=False, encoding='utf-8-sig')

def run_dual_modeling(processed_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    if os.path.exists(global_file):
        train_and_extract_insights(global_file, 'Global Platforms (جهانی)', models_dir)
    else:
        print("فایل داده‌های جهانی یافت نشد!")
        
    if os.path.exists(native_file):
        train_and_extract_insights(native_file, 'Native Platforms (بومی ایران)', models_dir)
    else:
        print("فایل داده‌های بومی یافت نشد!")
        
    print(f"\nپایان عملیات. خروجی‌های تحلیلی در مسیر {models_dir} ذخیره شدند.")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    run_dual_modeling(processed_dir, models_dir)