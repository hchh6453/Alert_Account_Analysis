#!/usr/bin/env python3
"""
特徵工程結果展示腳本
展示從交易資料中提取的特徵
"""

import pandas as pd
import numpy as np
from data_loader import DataLoader
from feature_engineering import FeatureEngineer

def show_feature_engineering_results():
    """展示特徵工程結果"""
    print("=== 特徵工程結果展示 ===\n")
    
    # 1. 載入資料
    print("1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    print(f"交易資料: {len(transactions_df):,} 筆")
    print(f"警示帳戶: {len(alerts_df):,} 個")
    print(f"預測目標: {len(predict_df):,} 個")
    
    # 2. 展示原始資料結構
    print(f"\n2. 交易資料欄位:")
    print(f"   {list(transactions_df.columns)}")
    
    print(f"\n3. 交易資料樣本:")
    print(transactions_df.head(3).to_string())
    
    # 3. 特徵工程
    print(f"\n4. 進行特徵工程...")
    feature_engineer = FeatureEngineer()
    
    # 只處理前100個帳戶作為示範
    sample_accounts = list(set(transactions_df['from_acct'].unique()) | set(transactions_df['to_acct'].unique()))[:100]
    
    print(f"處理 {len(sample_accounts)} 個帳戶的特徵...")
    
    # 建立特徵
    features_list = []
    for i, account in enumerate(sample_accounts):
        if i % 20 == 0:
            print(f"  處理進度: {i+1}/{len(sample_accounts)}")
        
        account_features = feature_engineer.create_single_account_features(
            account, transactions_df, alerts_df, predict_df
        )
        features_list.append(account_features)
    
    features_df = pd.DataFrame(features_list)
    
    # 4. 展示特徵結果
    print(f"\n5. 特徵工程結果:")
    print(f"   帳戶數量: {len(features_df)}")
    print(f"   特徵數量: {len(features_df.columns)}")
    
    # 5. 展示特徵類別
    print(f"\n6. 特徵類別分析:")
    
    feature_categories = {
        '時間特徵': [col for col in features_df.columns if any(keyword in col for keyword in ['transaction', 'night', 'weekend', 'interval', 'frequency'])],
        '金額特徵': [col for col in features_df.columns if any(keyword in col for keyword in ['amount', 'amt', 'large', 'inbound', 'outbound', 'net'])],
        '對手特徵': [col for col in features_df.columns if any(keyword in col for keyword in ['counterparty', 'unique', 'concentration', 'repeat', 'self'])],
        '模式特徵': [col for col in features_df.columns if any(keyword in col for keyword in ['burst', 'same', 'pattern', 'regular'])],
        '網路特徵': [col for col in features_df.columns if any(keyword in col for keyword in ['network', 'density', 'clustering'])],
        '其他特徵': [col for col in features_df.columns if col not in ['acct', 'label'] and not any(keyword in col for keyword in ['transaction', 'night', 'weekend', 'interval', 'frequency', 'amount', 'amt', 'large', 'inbound', 'outbound', 'net', 'counterparty', 'unique', 'concentration', 'repeat', 'self', 'burst', 'same', 'pattern', 'regular', 'network', 'density', 'clustering'])]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"   {category}: {len(features)} 個")
            print(f"     {features[:5]}{'...' if len(features) > 5 else ''}")
    
    # 6. 展示特徵統計
    print(f"\n7. 特徵統計摘要:")
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['label']]
    
    if len(numeric_features) > 0:
        stats_df = features_df[numeric_features].describe()
        print(f"   數值特徵統計 (前10個):")
        print(stats_df.head(10).round(4).to_string())
    
    # 7. 展示標籤分佈
    print(f"\n8. 標籤分佈:")
    label_counts = features_df['label'].value_counts(dropna=False)
    print(f"   正常帳戶 (0): {label_counts.get(0, 0)}")
    print(f"   警示帳戶 (1): {label_counts.get(1, 0)}")
    print(f"   無標籤 (NaN): {features_df['label'].isna().sum()}")
    
    # 8. 展示高風險特徵
    print(f"\n9. 高風險帳戶特徵分析:")
    alert_accounts = features_df[features_df['label'] == 1]
    normal_accounts = features_df[features_df['label'] == 0]
    
    if len(alert_accounts) > 0 and len(normal_accounts) > 0:
        print(f"   警示帳戶平均特徵值 (前10個):")
        alert_means = alert_accounts[numeric_features].mean().sort_values(ascending=False)
        print(f"   {alert_means.head(10).round(4).to_string()}")
        
        print(f"\n   正常帳戶平均特徵值 (前10個):")
        normal_means = normal_accounts[numeric_features].mean().sort_values(ascending=False)
        print(f"   {normal_means.head(10).round(4).to_string()}")
    
    # 9. 儲存特徵結果
    print(f"\n10. 儲存特徵結果...")
    features_df.to_csv('feature_engineering_results.csv', index=False)
    print(f"   特徵結果已儲存至: feature_engineering_results.csv")
    
    print(f"\n=== 特徵工程展示完成 ===")

if __name__ == "__main__":
    show_feature_engineering_results()
