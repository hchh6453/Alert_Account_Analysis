"""
分析警示帳戶的特徵模式
找出異常帳戶的關鍵特徵
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import FeatureEngineer
from data_loader import DataLoader

def analyze_alert_features():
    """分析警示帳戶的特徵模式"""
    print("=== 警示帳戶特徵分析 ===")
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    # 限制資料量
    transactions_df = transactions_df.head(100000)  # 10萬筆交易
    
    # 2. 建立特徵
    print("\n2. 建立特徵...")
    feature_engineer = FeatureEngineer()
    
    # 只為警示帳戶和部分正常帳戶建立特徵
    alert_accounts = set(alerts_df['acct'])
    from_accounts = set(transactions_df['from_acct'].unique())
    to_accounts = set(transactions_df['to_acct'].unique())
    all_accounts = list(from_accounts.union(to_accounts))
    
    # 取樣：所有警示帳戶 + 1000個正常帳戶
    normal_accounts = [acc for acc in all_accounts if acc not in alert_accounts]
    sample_accounts = list(alert_accounts) + normal_accounts[:1000]
    
    print(f"分析帳戶: {len(sample_accounts)} 個")
    print(f"其中警示帳戶: {len(alert_accounts)} 個")
    
    # 建立特徵
    features_list = []
    for i, account in enumerate(sample_accounts):
        if i % 100 == 0:
            print(f"  處理進度: {i+1}/{len(sample_accounts)}")
        
        account_features = feature_engineer.create_single_account_features(
            account, transactions_df, alerts_df, predict_df
        )
        features_list.append(account_features)
    
    features_df = pd.DataFrame(features_list)
    
    # 3. 分析特徵差異
    print("\n3. 分析特徵差異...")
    
    alert_features = features_df[features_df['label'] == 1]
    normal_features = features_df[features_df['label'] == 0]
    
    print(f"警示帳戶樣本: {len(alert_features)} 個")
    print(f"正常帳戶樣本: {len(normal_features)} 個")
    
    # 計算特徵差異
    feature_cols = [col for col in features_df.columns if col not in ['acct', 'label']]
    
    feature_analysis = []
    for col in feature_cols:
        alert_mean = alert_features[col].mean()
        normal_mean = normal_features[col].mean()
        
        # 計算差異倍數
        if normal_mean != 0:
            ratio = alert_mean / normal_mean
        else:
            ratio = float('inf') if alert_mean > 0 else 1
        
        feature_analysis.append({
            'feature': col,
            'alert_mean': alert_mean,
            'normal_mean': normal_mean,
            'ratio': ratio,
            'abs_ratio': abs(ratio - 1)
        })
    
    analysis_df = pd.DataFrame(feature_analysis)
    analysis_df = analysis_df.sort_values('abs_ratio', ascending=False)
    
    # 4. 顯示最重要的特徵
    print("\n4. 最重要的特徵差異 (前20個):")
    print("=" * 80)
    for i, row in analysis_df.head(20).iterrows():
        print(f"{row['feature']:30} | 警示: {row['alert_mean']:8.2f} | 正常: {row['normal_mean']:8.2f} | 倍數: {row['ratio']:6.2f}")
    
    # 5. 保存分析結果
    analysis_df.to_csv('feature_analysis.csv', index=False)
    print(f"\n分析結果已保存至 feature_analysis.csv")
    
    # 6. 建議新特徵
    print("\n5. 建議的新特徵:")
    suggest_new_features(analysis_df, alert_features, normal_features)
    
    return analysis_df

def suggest_new_features(analysis_df, alert_features, normal_features):
    """建議新的特徵"""
    
    print("\n基於分析結果，建議以下新特徵:")
    print("=" * 50)
    
    # 1. 異常值檢測特徵
    print("1. 異常值檢測特徵:")
    print("   - 交易金額的Z-score")
    print("   - 交易頻率的Z-score") 
    print("   - 交易時間間隔的Z-score")
    
    # 2. 行為模式特徵
    print("\n2. 行為模式特徵:")
    print("   - 交易金額的變異係數 (CV)")
    print("   - 交易時間的規律性指標")
    print("   - 交易對手的多樣性指標")
    
    # 3. 時間序列特徵
    print("\n3. 時間序列特徵:")
    print("   - 交易趨勢 (上升/下降)")
    print("   - 交易週期性檢測")
    print("   - 交易突發性檢測")
    
    # 4. 網絡特徵
    print("\n4. 網絡特徵:")
    print("   - 帳戶的PageRank分數")
    print("   - 帳戶的聚類係數")
    print("   - 帳戶的介數中心性")
    
    # 5. 組合特徵
    print("\n5. 組合特徵:")
    print("   - 金額 × 頻率")
    print("   - 夜間交易 × 大額交易")
    print("   - 新對手 × 高頻交易")

if __name__ == "__main__":
    analyze_alert_features()
