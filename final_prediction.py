#!/usr/bin/env python3
"""
完整預測腳本 - 處理所有資料並輸出符合 submission_template 格式的結果
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
import os
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from anomaly_detector import AnomalyDetector

def create_final_submission():
    """建立最終的預測結果"""
    print("=== 建立最終預測結果 ===")
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    print(f"交易資料: {len(transactions_df):,} 筆")
    print(f"警示帳戶: {len(alerts_df):,} 個")
    print(f"預測目標: {len(predict_df):,} 個")
    
    # 2. 讀取 submission_template
    print("\n2. 讀取 submission_template...")
    template_df = pd.read_csv('submission_template.csv')
    print(f"Template 包含 {len(template_df)} 個帳戶")
    
    # 3. 特徵工程 (只處理預測目標帳戶)
    print("\n3. 進行特徵工程...")
    feature_engineer = FeatureEngineer()
    
    # 只為預測目標帳戶建立特徵
    predict_accounts = predict_df['acct'].tolist()
    print(f"需要處理 {len(predict_accounts)} 個預測帳戶")
    
    # 分批處理以避免記憶體問題
    batch_size = 1000
    all_features = []
    
    for i in range(0, len(predict_accounts), batch_size):
        batch_accounts = predict_accounts[i:i+batch_size]
        print(f"  處理批次 {i//batch_size + 1}/{(len(predict_accounts)-1)//batch_size + 1}")
        
        batch_features = []
        for account in batch_accounts:
            account_features = feature_engineer.create_single_account_features(
                account, transactions_df, alerts_df, predict_df
            )
            batch_features.append(account_features)
        
        all_features.extend(batch_features)
    
    account_features = pd.DataFrame(all_features)
    
    print(f"建立特徵: {len(account_features):,} 個帳戶")
    print(f"特徵維度: {account_features.shape[1]} 個特徵")
    
    # 4. 準備訓練資料 (使用警示帳戶作為正樣本)
    print("\n4. 準備訓練資料...")
    
    # 為警示帳戶建立特徵
    alert_accounts = alerts_df['acct'].tolist()[:500]  # 限制數量以避免記憶體問題
    
    print(f"為 {len(alert_accounts)} 個警示帳戶建立特徵...")
    alert_features = []
    for account in alert_accounts:
        account_features = feature_engineer.create_single_account_features(
            account, transactions_df, alerts_df, predict_df
        )
        alert_features.append(account_features)
    
    alert_features_df = pd.DataFrame(alert_features)
    
    # 合併訓練資料
    train_features = pd.concat([account_features, alert_features_df], ignore_index=True)
    
    # 分離特徵和標籤
    feature_cols = [col for col in train_features.columns if col not in ['acct', 'label']]
    X = train_features[feature_cols]
    y = train_features['label']
    
    print(f"訓練資料: {len(X)} 筆")
    print(f"特徵數: {len(feature_cols)} 個")
    
    # 5. 訓練模型
    print("\n5. 訓練模型...")
    detector = AnomalyDetector()
    
    # 使用 Isolation Forest 進行異常檢測
    model = detector.train_isolation_forest(X)
    
    # 6. 預測
    print("\n6. 進行預測...")
    
    # 只對預測目標帳戶進行預測
    predict_features = account_features[feature_cols]
    predictions = model.predict(predict_features)
    scores = model.decision_function(predict_features)
    
    # 轉換為 0/1 標籤 (-1 -> 1, 1 -> 0)
    predictions = np.where(predictions == -1, 1, 0)
    
    # 7. 建立符合 submission_template 格式的結果
    print("\n7. 建立預測結果...")
    
    # 建立預測結果 DataFrame
    submission_df = pd.DataFrame({
        'acct': account_features['acct'],
        'label': predictions.astype(int)
    })
    
    # 確保所有 template 中的帳戶都在結果中
    missing_accounts = set(template_df['acct']) - set(submission_df['acct'])
    if missing_accounts:
        print(f"為 {len(missing_accounts)} 個缺失帳戶添加預設預測...")
        missing_df = pd.DataFrame({
            'acct': list(missing_accounts),
            'label': [0] * len(missing_accounts)  # 預設為正常帳戶
        })
        submission_df = pd.concat([submission_df, missing_df], ignore_index=True)
    
    # 確保順序與 template 一致
    submission_df = submission_df.set_index('acct').reindex(template_df['acct']).reset_index()
    
    # 儲存結果
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"預測結果已儲存至 submission.csv")
    print(f"預測警示帳戶數量: {predictions.sum():,}")
    print(f"預測正常帳戶數量: {(predictions == 0).sum():,}")
    
    # 驗證結果格式
    print(f"\n8. 驗證結果格式...")
    print(f"結果檔案行數: {len(submission_df)}")
    print(f"Template 行數: {len(template_df)}")
    print(f"格式是否一致: {len(submission_df) == len(template_df)}")
    
    # 檢查欄位格式
    print(f"結果欄位: {list(submission_df.columns)}")
    print(f"Template 欄位: {list(template_df.columns)}")
    print(f"欄位是否一致: {list(submission_df.columns) == list(template_df.columns)}")
    
    # 統計預測分佈
    label_counts = submission_df['label'].value_counts()
    print(f"\n預測分佈:")
    print(f"正常帳戶 (0): {label_counts.get(0, 0):,}")
    print(f"警示帳戶 (1): {label_counts.get(1, 0):,}")
    
    # 顯示前幾行結果
    print(f"\n前5行預測結果:")
    print(submission_df.head())
    
    # 同時儲存詳細結果
    detailed_df = pd.DataFrame({
        'acct': account_features['acct'],
        'label': predictions.astype(int),
        'score': scores
    })
    detailed_df.to_csv('detailed_predictions.csv', index=False)
    print(f"詳細預測結果已儲存至 detailed_predictions.csv")
    
    print(f"\n=== 預測完成 ===")
    print(f"主要輸出檔案: submission.csv")
    print(f"格式: 符合 submission_template.csv")

if __name__ == "__main__":
    create_final_submission()
