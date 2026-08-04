import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def clean_version(version_str):
    try:
        v_str = str(version_str).strip()
        if pd.isna(v_str) or 'unknown' in v_str.lower():
            return None
        match = re.search(r'^(\d+)\.(\d+)', v_str)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            if major > 100 or minor > 999: return None
            return float(f"{major}.{minor}")
        return None
    except: return None

def process_and_plot_stage_grid(filepath, ecosystem_name, color_hex, output_dir):
    print(f"✨ پردازش ماتریس استانداردشده (Stage-based) برای: {ecosystem_name}...")
    
    df = pd.read_csv(filepath)
    if not all(col in df.columns for col in ['app_version', 'Composite_Trust', 'app_label']): return
        
    df = df.dropna(subset=['app_version', 'Composite_Trust', 'app_label'])
    df['Clean_Version'] = df['app_version'].apply(clean_version)
    df = df.dropna(subset=['Clean_Version'])
    
    # فیلتر آستانه (حداقل 15 کامنت برای اعتبار آماری)
    version_counts = df.groupby(['app_label', 'Clean_Version']).size().reset_index(name='count')
    df = pd.merge(df, version_counts[version_counts['count'] >= 15][['app_label', 'Clean_Version']], on=['app_label', 'Clean_Version'])
    
    # انتخاب 6 اپلیکیشن برتر
    top_6_apps = df['app_label'].value_counts().nlargest(6).index
    df = df[df['app_label'].isin(top_6_apps)]
    if df.empty: return
    
    trust_trend = df.groupby(['app_label', 'Clean_Version'])['Composite_Trust'].mean() * 100
    trust_trend = trust_trend.reset_index()
    
    # نگه داشتن 10 نسخه برتر هر اپلیکیشن
    trust_trend = trust_trend.groupby('app_label', group_keys=False).apply(lambda g: g.sort_values('Clean_Version').tail(10))
    
    # === جادوی اصلی: تبدیل تمام نسخه‌ها به Stage ===
    trust_trend['Stage_Num'] = trust_trend.groupby('app_label')['Clean_Version'].rank(method='dense').astype(int)
    trust_trend['Display_Version'] = 'Stage ' + trust_trend['Stage_Num'].astype(str)
    
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fbfcfc", "grid.linestyle": "--"})
    g = sns.relplot(
        data=trust_trend, x='Display_Version', y='Composite_Trust',
        col='app_label', col_wrap=3, kind='line', marker='o', color=color_hex,
        facet_kws={'sharex': False, 'sharey': True}, height=4, aspect=1.3, linewidth=3, markersize=8
    )
    
    g.set_titles(col_template="{col_name}", size=14, weight='bold')
    g.set_axis_labels("Normalized Release Stage (1 to N)", "Average Trust (%)", weight='bold')
    g.set(ylim=(0, 100))
    
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        lines = ax.get_lines()
        if lines:
            ax.fill_between(lines[0].get_xdata(), lines[0].get_ydata(), color=color_hex, alpha=0.1)
            
    plt.subplots_adjust(top=0.88, hspace=0.5)
    g.fig.suptitle(f"{ecosystem_name}: Trust Lifecycle Across Standardized Stages", fontsize=18, fontweight='bold')
    
    safe_name = ecosystem_name.replace(' ', '_').replace('/', '_')
    save_path = os.path.join(output_dir, f'Stage_Grid_{safe_name}.png')
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ نمودار Stage ذخیره شد: {save_path}")

if __name__ == "__main__":
    base_dir = '..'
    global_file = os.path.join(base_dir, 'data', 'processed', 'ALL_GLOBAL_BALANCED_ready_for_ml.csv')
    native_file = os.path.join(base_dir, 'data', 'processed', 'ALL_NATIVE_BALANCED_ready_for_ml.csv')
    out_dir = os.path.join(base_dir, 'models')
    
    if os.path.exists(global_file): process_and_plot_stage_grid(global_file, 'Global AI Platforms', '#1f4e79', out_dir)
    if os.path.exists(native_file): process_and_plot_stage_grid(native_file, 'Native AI Platforms', '#a93226', out_dir)