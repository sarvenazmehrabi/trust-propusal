import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import os

def plot_true_confusion_matrix(filepath, ecosystem_name, output_dir, text_col='final_clean_content', target_col='Composite_Trust'):
    print(f"\n🧠 در حال پردازش و رسم ماتریس نهایی برای: {ecosystem_name}...")
    
    df = pd.read_csv(filepath)
    df = df.dropna(subset=[text_col, target_col])
    
    # 🔴 کلید حل مشکل: تبدیل 0 و 1 به نام‌های متنی برای جلوگیری از باگ Scikit-Learn
    mapping = {
        '0': 'Low Trust (Distrust)', '1': 'High Trust', 
        0: 'Low Trust (Distrust)', 1: 'High Trust',
        0.0: 'Low Trust (Distrust)', 1.0: 'High Trust'
    }
    df['Target_Class'] = df[target_col].map(mapping)
    df['Target_Class'] = df['Target_Class'].fillna(df[target_col].astype(str))
    
    y = df['Target_Class']
    X = df[text_col]
    
    labels_order = ['Low Trust (Distrust)', 'High Trust']
    print(f"✅ کلاس‌های نگاشت شده برای مدل: {list(y.unique())}")
    
    # برداری‌سازی
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)
    
    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
    
    # آموزش مدل
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # پیش‌بینی
    y_pred = rf_model.predict(X_test)
    
    # محاسبه ماتریس درهم‌ریختگی
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    
    # محاسبه درصدها
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_sum[cm_sum == 0] = 1 # جلوگیری از خطای تقسیم بر صفر
    cm_percentage = cm.astype('float') / cm_sum * 100
    
    # رسم ماتریس
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    
    labels = [f"{v1}\n({v2:.1f}%)" for v1, v2 in zip(cm.flatten(), cm_percentage.flatten())]
    labels = np.asarray(labels).reshape(2, 2)
    
    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True,
                     xticklabels=labels_order, yticklabels=labels_order,
                     annot_kws={"size": 12, "weight": "bold"})
    
    plt.title(f'Confusion Matrix: Random Forest Model\n{ecosystem_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontsize=12, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    
    safe_name = ecosystem_name.replace(' ', '_').replace('/', '_')
    save_path = os.path.join(output_dir, f'Confusion_Matrix_Final_{safe_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ماتریس درهم‌ریختگی با موفقیت ذخیره شد: {save_path}")

if __name__ == "__main__":
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    global_file = os.path.join(processed_dir, 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(processed_dir, 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    
    print(f"{'='*60}")
    if os.path.exists(global_file): 
        plot_true_confusion_matrix(global_file, 'Global AI Platforms', models_dir)
    if os.path.exists(native_file): 
        plot_true_confusion_matrix(native_file, 'Native AI Platforms', models_dir)
    print(f"{'='*60}\n")