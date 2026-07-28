import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot style for academic reports
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

def run_eda():
    # 1. Define paths
    data_dir = r"E:\projects\mehrabi\codes\data_ingestion\data"
    output_dir = r"E:\projects\mehrabi\codes\eda_outputs"
    
    # Create output directories for reports and plots
    reports_dir = os.path.join(output_dir, "reports")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("[INFO] Starting Exploratory Data Analysis (EDA)...")

    # 2. Load and aggregate data
    all_data = []
    
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.csv'):
            continue
            
        file_path = os.path.join(data_dir, file_name)
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # Determine platform type based on file name prefix
            if file_name.startswith("balanced_data"):
                df['platform_type'] = 'Global'
            elif file_name.startswith("native_data"):
                df['platform_type'] = 'Native'
            else:
                df['platform_type'] = 'Unknown'
                
            all_data.append(df)
            print(f"[INFO] Loaded: {file_name} | Shape: {df.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load {file_name}: {e}")

    # Combine all dataframes into a single Master DataFrame
    master_df = pd.concat(all_data, ignore_index=True)
    print(f"\n[INFO] Master DataFrame created. Total records: {len(master_df)}")

    # 3. Generate Statistical Tables (For Word/Proposal)
    
    # Table A: Basic stats per App (Count, Avg Score)
    # Note: Assuming 'app_label' and 'score' columns exist based on previous reports
    if 'app_label' in master_df.columns and 'score' in master_df.columns:
        app_stats = master_df.groupby(['platform_type', 'app_label']).agg(
            total_comments=('content', 'count'),
            avg_score=('score', 'mean')
        ).reset_index()
        
        app_stats['avg_score'] = app_stats['avg_score'].round(2)
        app_stats.to_csv(os.path.join(reports_dir, "app_basic_stats.csv"), index=False, encoding='utf-8-sig')
        print("[INFO] Saved: app_basic_stats.csv")

    # Table B: Missing Values (Null Check)
    null_stats = master_df.isnull().sum().to_frame(name='null_count')
    null_stats['null_percentage'] = (null_stats['null_count'] / len(master_df) * 100).round(2)
    null_stats.to_csv(os.path.join(reports_dir, "null_values_report.csv"), encoding='utf-8-sig')
    print("[INFO] Saved: null_values_report.csv")

    # Table C: Developer Reply Rate (Global vs Native)
    if 'developer_reply' in master_df.columns:
        # Check if developer_reply is not null
        master_df['has_reply'] = master_df['developer_reply'].notna()
        
        reply_stats = master_df.groupby('platform_type').agg(
            total_reviews=('content', 'count'),
            replied_reviews=('has_reply', 'sum')
        ).reset_index()
        
        reply_stats['reply_rate_percentage'] = (reply_stats['replied_reviews'] / reply_stats['total_reviews'] * 100).round(2)
        reply_stats.to_csv(os.path.join(reports_dir, "developer_reply_rate.csv"), index=False, encoding='utf-8-sig')
        print("[INFO] Saved: developer_reply_rate.csv")

    # 4. Generate Visualizations (For Proposal/Defense)
    
    # Plot 1: Total Comments per Platform Type
    plt.figure(figsize=(8, 6))
    sns.countplot(data=master_df, x='platform_type', palette='viridis')
    plt.title('Total Number of Reviews: Global vs Native Platforms')
    plt.ylabel('Number of Reviews')
    plt.xlabel('Platform Type')
    plt.savefig(os.path.join(plots_dir, "volume_comparison.png"), dpi=300)
    plt.close()
    print("[INFO] Generated Plot: volume_comparison.png")

    # Plot 2: Average Score per App
    if 'app_label' in master_df.columns:
        plt.figure(figsize=(12, 6))
        order = master_df.groupby('app_label')['score'].mean().sort_values(ascending=False).index
        sns.barplot(data=master_df, x='app_label', y='score', hue='platform_type', order=order, palette='muted', dodge=False)
        plt.title('Average Score per Chatbot')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Star Rating')
        plt.xlabel('Chatbot Name')
        plt.legend(title='Platform Type')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "avg_score_per_app.png"), dpi=300)
        plt.close()
        print("[INFO] Generated Plot: avg_score_per_app.png")

    # Plot 3: Developer Reply Rate Comparison
    if 'has_reply' in master_df.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=reply_stats, x='platform_type', y='reply_rate_percentage', palette='magma')
        plt.title('Developer Reply Rate (%)')
        plt.ylabel('Reply Rate (%)')
        plt.xlabel('Platform Type')
        
        # Add percentage labels on top of bars
        for index, row in reply_stats.iterrows():
            plt.text(index, row.reply_rate_percentage + 0.5, f"{row.reply_rate_percentage}%", color='black', ha="center")
            
        plt.savefig(os.path.join(plots_dir, "reply_rate_comparison.png"), dpi=300)
        plt.close()
        print("[INFO] Generated Plot: reply_rate_comparison.png")

    print(f"\n[SUCCESS] EDA complete. Reports and plots are saved in: {output_dir}")

if __name__ == "__main__":
    # Ensure proper library installations: pip install pandas matplotlib seaborn
    run_eda()