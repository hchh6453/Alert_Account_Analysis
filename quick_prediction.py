#!/usr/bin/env python3
"""
快速預測腳本 - 使用樣本資料快速測試預測流程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from anomaly_detector import AnomalyDetector

def quick_prediction():
    """快速預測測試"""
    print("=== 快速預測測試 ===")
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    # 限制資料量以快速測試
    sample_size = 1000
    transactions_df = transactions_df.head(sample_size * 10)  # 取更多交易以確保有足夠的帳戶
    
    print(f"交易資料 (樣本): {len(transactions_df):,} 筆")
    print(f"警示帳戶: {len(alerts_df):,} 個")
    print(f"預測目標: {len(predict_df):,} 個")
    
    # 2. 特徵工程 (限制帳戶數量)
    print("\n2. 進行特徵工程...")
    feature_engineer = FeatureEngineer()
    
    # 取得樣本帳戶
    from_accounts = set(transactions_df['from_acct'].unique())
    to_accounts = set(transactions_df['to_acct'].unique())
    sample_accounts = list(from_accounts.union(to_accounts))[:sample_size]
    
    # 為樣本帳戶建立特徵
    features_list = []
    for i, account in enumerate(sample_accounts):
        if i % 100 == 0:
            print(f"  處理進度: {i+1}/{len(sample_accounts)}")
        
        account_features = feature_engineer.create_single_account_features(
            account, transactions_df, alerts_df, predict_df
        )
        features_list.append(account_features)
    
    account_features = pd.DataFrame(features_list)
    
    print(f"建立特徵: {len(account_features):,} 個帳戶")
    print(f"特徵維度: {account_features.shape[1]} 個特徵")
    
    # 3. 準備訓練和測試資料
    print("\n3. 準備訓練和測試資料...")
    
    # 分離有標籤和無標籤的資料
    labeled_data = account_features[account_features['label'].notna()]
    unlabeled_data = account_features[account_features['label'].isna()]
    
    print(f"有標籤資料: {len(labeled_data):,} 筆")
    print(f"無標籤資料: {len(unlabeled_data):,} 筆")
    
    if len(labeled_data) < 10:
        print("警告: 有標籤資料太少，無法進行有效訓練")
        return
    
    # 分離特徵和標籤
    feature_cols = [col for col in account_features.columns if col not in ['acct', 'label']]
    X = labeled_data[feature_cols]
    y = labeled_data['label']
    
    # 分割訓練和驗證資料
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練資料: {len(X_train):,} 筆")
    print(f"驗證資料: {len(X_val):,} 筆")
    
    # 4. 訓練模型
    print("\n4. 訓練模型...")
    detector = AnomalyDetector()
    
    # 只使用 Random Forest 進行快速測試
    model = detector.train_random_forest(X_train, y_train)
    
    # 5. 模型評估
    print("\n5. 評估模型效能...")
    val_predictions = model.predict(X_val)
    val_scores = model.predict_proba(X_val)[:, 1]
    f1 = f1_score(y_val, val_predictions)
    
    print(f"F1-Score: {f1:.4f}")
    
    # 6. 預測
    print("\n6. 進行預測...")
    if len(unlabeled_data) > 0:
        X_unlabeled = unlabeled_data[feature_cols]
        predictions = model.predict(X_unlabeled)
        scores = model.predict_proba(X_unlabeled)[:, 1]
        
        # 建立結果
        result_df = pd.DataFrame({
            'acct': unlabeled_data['acct'],
            'label': predictions.astype(int)
        })
        
        print(f"預測警示帳戶數量: {predictions.sum():,}")
        print(f"預測正常帳戶數量: {(predictions == 0).sum():,}")
        
        # 儲存結果
        result_df.to_csv('quick_submission.csv', index=False)
        print(f"快速預測結果已儲存至 quick_submission.csv")
        
        # 顯示前幾行
        print(f"\n前5行預測結果:")
        print(result_df.head())
    else:
        print("沒有需要預測的帳戶")
    
    print(f"\n=== 快速預測完成 ===")

if __name__ == "__main__":
    quick_prediction()
