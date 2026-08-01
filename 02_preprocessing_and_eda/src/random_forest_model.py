import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def prepare_data(file_list):
    """
    آماده‌سازی و مهندسی ویژگی‌ها برای ورود به مدل
    """
    df_list = []
    for f in file_list:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    df = pd.concat(df_list, ignore_index=True)
    
    # 1. حذف نویزها (چون نویزها جنبه خاصی ندارند و فقط مدل را گیج می‌کنند)
    df = df[~df['Final_Aspect'].isin(['Noise', 'شاخص‌های رفتاری'])].copy()
    
    # 2. حذف رکوردهایی که اسکور (ستاره) ندارند
    df = df.dropna(subset=['score', 'Sentiment_Polarity', 'Final_Aspect'])
    
    # 3. مهندسی ویژگی: ضربِ One-Hot در قطبیت احساسات
    # با این کار، اگر کاربر درباره سرعت مثبت حرف زده باشد، مقدار آن ستون 1 می شود، اگر منفی حرف زده باشد 1-
    features = pd.get_dummies(df['Final_Aspect'], dtype=float)
    for col in features.columns:
        features[col] = features[col] * df['Sentiment_Polarity']
        
    # متغیر هدف (Y) همان امتیاز 1 تا 5 ستاره کاربر است
    X = features
    y = df['score'].astype(int)
    
    return X, y

def train_and_extract_importance(X, y, model_name, output_dir):
    """
    آموزش جنگل تصادفی و استخراج میزان اهمیت شاخص‌ها (Feature Importance)
    """
    print(f"\n--- در حال آموزش مدل {model_name} ---")
    print(f"تعداد رکوردهای معتبر (بدون نویز): {len(X)}")
    
    if len(X) == 0:
        print("داده‌ای برای آموزش وجود ندارد!")
        return

    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تعریف و آموزش مدل Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # دقت مدل
    accuracy = rf.score(X_test, y_test)
    print(f"دقت پیش‌بینی مدل (Accuracy): {accuracy:.2f}")
    
    # استخراج اهمیت ویژگی‌ها (Feature Importance)
    importances = rf.feature_importances_
    
    # مرتب‌سازی و ذخیره
    feat_imp_df = pd.DataFrame({
        'Aspect': X.columns,
        'Importance_Score': importances
    }).sort_values(by='Importance_Score', ascending=False)
    
    print("\nرتبه‌بندی شاخص‌های اعتماد:")
    print(feat_imp_df.to_string(index=False))
    
    # ذخیره در فایل CSV
    feat_imp_df.to_csv(os.path.join(output_dir, f'{model_name}_feature_importance.csv'), index=False, encoding='utf-8-sig')

def run_ml_pipeline(processed_dir, models_dir):
    # تفکیک فایل‌های جهانی (Global) و بومی (Native)
    all_files = glob.glob(os.path.join(processed_dir, '*_ready_for_ml.csv'))
    
    global_files = [f for f in all_files if 'native_data' not in f]
    native_files = [f for f in all_files if 'native_data' in f]
    
    print(">>> شروع استخراج الگوهای پلتفرم‌های جهانی (Global Platforms) ...")
    X_glob, y_glob = prepare_data(global_files)
    train_and_extract_importance(X_glob, y_glob, 'Global_Platforms', models_dir)
    
    print("\n======================================================\n")
    
    print(">>> شروع استخراج الگوهای پلتفرم‌های بومی ایران (Native Platforms) ...")
    X_nat, y_nat = prepare_data(native_files)
    train_and_extract_importance(X_nat, y_nat, 'Native_Platforms', models_dir)
    
    print(f"\nپایان عملیات. فایل‌های تحلیل اهمیت در مسیر {models_dir} ذخیره شدند.")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    run_ml_pipeline(processed_dir, models_dir)