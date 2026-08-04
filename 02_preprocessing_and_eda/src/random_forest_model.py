import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib  # برای ذخیره مدل‌ها جهت انتشار در گیت‌هاب
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def prepare_data(df):
    df = df[df['Composite_Trust'] != -1].copy()
    df = df.dropna(subset=['Composite_Trust', 'Final_Aspect'])
    X = pd.get_dummies(df['Final_Aspect'], dtype=float)
    y = df['Composite_Trust'].astype(int)
    
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
    X = X.rename(columns=aspect_mapping)
    return X, y

def train_and_evaluate(filepath, platform_name, output_dir):
    print(f"\n{'='*50}")
    print(f"🚀 آموزش مدل Random Forest برای: {platform_name}")
    print(f"{'='*50}")
    
    df = pd.read_csv(filepath)
    X, y = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,            # تنظیم عمق برای داده‌های ساده‌تر
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # 💾 ذخیره مدل برای گیت‌هاب
    safe_name = platform_name.replace(' ', '_').replace('/', '_')
    model_path = os.path.join(output_dir, f'RF_Model_{safe_name}.pkl')
    joblib.dump(rf, model_path)
    print(f"💾 فایل مدل برای گیت‌هاب ذخیره شد: {model_path}")
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ دقت مدل (Accuracy): {acc * 100:.2f}%")
    
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Trust_Aspect': X.columns,
        'Importance_Weight': importances
    }).sort_values(by='Importance_Weight', ascending=False)
    
    feat_imp_df['Importance (%)'] = (feat_imp_df['Importance_Weight'] * 100).round(2)
    print(f"\n📊 رتبه‌بندی اهمیت شاخص‌ها در {platform_name}:")
    print(feat_imp_df[['Trust_Aspect', 'Importance (%)']].to_string(index=False))
    
    plot_feature_importance(feat_imp_df, platform_name, output_dir)

def plot_feature_importance(feat_imp_df, platform_name, output_dir):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # اصلاح هشدار Seaborn با اضافه کردن hue
    ax = sns.barplot(
        data=feat_imp_df, 
        x='Importance (%)', 
        y='Trust_Aspect', 
        hue='Trust_Aspect',
        palette='viridis',
        legend=False
    )
    
    plt.title(f'Random Forest Feature Importance - {platform_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Impact on User Trust (%)', fontsize=12)
    plt.ylabel('Trust Categories', fontsize=12)
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(width + 0.5, p.get_y() + p.get_height()/2. + 0.1, f'{width:.1f}%', va='center')
        
    plt.tight_layout()
    safe_name = platform_name.replace(' ', '_').replace('/', '_')
    save_path = os.path.join(output_dir, f'RF_Importance_{safe_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    if os.path.exists(global_file):
        train_and_evaluate(global_file, 'Global AI Platforms', models_dir)
        
    if os.path.exists(native_file):
        train_and_evaluate(native_file, 'Native AI Platforms (Iran)', models_dir)