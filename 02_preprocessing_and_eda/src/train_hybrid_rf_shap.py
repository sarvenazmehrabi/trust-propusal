import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

def train_hybrid_model(filepath, platform_name, output_dir):
    print(f"\n{'='*60}")
    print(f"🚀 آموزش مدل ترکیبی (TF-IDF + Structural) برای: {platform_name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(filepath)
    TEXT_COLUMN = 'final_clean_content'
    
    if TEXT_COLUMN not in df.columns:
        print(f"❌ خطا: ستون '{TEXT_COLUMN}' یافت نشد.")
        return
        
    df = df[df['Composite_Trust'] != -1].copy()
    df = df.dropna(subset=['Composite_Trust', 'Final_Aspect', TEXT_COLUMN])
    
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
    X_struct = pd.get_dummies(df['Final_Aspect'], dtype=float).rename(columns=aspect_mapping)
    struct_features = list(X_struct.columns)
    
    print("در حال برداری‌سازی متن کامنت‌ها (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_text_sparse = tfidf.fit_transform(df[TEXT_COLUMN].astype(str))
    X_text = pd.DataFrame(X_text_sparse.toarray(), columns=tfidf.get_feature_names_out(), index=df.index)
    
    X_combined = pd.concat([X_struct, X_text], axis=1)
    y = df['Composite_Trust'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("در حال آموزش مدل Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ دقت پیش‌بینی مدل (Accuracy): {acc * 100:.2f}%")
    
    print("در حال محاسبه مقادیر SHAP برای تفسیر مدل...")
    explainer = shap.TreeExplainer(rf)
    X_test_sample = shap.sample(X_test, 500)
    shap_values = explainer.shap_values(X_test_sample)
    
    # استخراج مقادیر مربوط به کلاس 1 (اعتماد)
    shap_vals_trust = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # حل مشکل بُعدهای اضافه (در صورت بازگرداندن تعاملات)
    if len(shap_vals_trust.shape) > 2:
        shap_vals_trust = shap_vals_trust.sum(axis=2)
        
    struct_indices = [X_test_sample.columns.get_loc(col) for col in struct_features]
    shap_struct_only = shap_vals_trust[:, struct_indices]
    
    # محاسبه میانگین قدر مطلق اثر هر ویژگی
    mean_shap = np.abs(shap_struct_only).mean(axis=0)
    
    shap_df = pd.DataFrame({
        'Trust_Aspect': struct_features,
        'SHAP_Impact': mean_shap
    }).sort_values(by='SHAP_Impact', ascending=False)
    
    # نرمال‌سازی به درصد برای درک بهتر در رساله
    shap_df['Impact_Percentage'] = (shap_df['SHAP_Impact'] / shap_df['SHAP_Impact'].sum()) * 100
    
    print(f"\n📊 رتبه‌بندی اهمیت شاخص‌ها (بر اساس SHAP) در {platform_name}:")
    print(shap_df[['Trust_Aspect', 'Impact_Percentage']].round(2).to_string(index=False))
    
    # رسم نمودار استاندارد و ژورنالی
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=shap_df, x='Impact_Percentage', y='Trust_Aspect', hue='Trust_Aspect', palette='mako', legend=False)
    
    plt.title(f'Impact of Structural Features on User Trust (SHAP Analysis)\n{platform_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Relative Importance based on SHAP values (%)', fontsize=12)
    plt.ylabel('Trust Categories', fontsize=12)
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if width > 0:
            plt.text(width + 0.3, p.get_y() + p.get_height()/2. + 0.1, f'{width:.1f}%', va='center', fontsize=10)
            
    plt.tight_layout()
    safe_name = platform_name.replace(' ', '_').replace('/', '_')
    save_path = os.path.join(output_dir, f'SHAP_BarChart_{safe_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🖼️ نمودار نهایی و استاندارد SHAP ذخیره شد: {save_path}")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    if os.path.exists(global_file):
        train_hybrid_model(global_file, 'Global AI Platforms', models_dir)
        
    if os.path.exists(native_file):
        train_hybrid_model(native_file, 'Native AI Platforms (Iran)', models_dir)