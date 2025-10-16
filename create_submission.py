#!/usr/bin/env python3
"""
預測腳本 - 輸出符合 submission_template 格式的結果
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from anomaly_detector import AnomalyDetector
from evaluator import ModelEvaluator

def create_submission():
    """建立符合 submission_template 格式的預測結果"""
    print("=== 建立預測結果 ===")
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    print(f"交易資料: {len(transactions_df):,} 筆")
    print(f"警示帳戶: {len(alerts_df):,} 個")
    print(f"預測目標: {len(predict_df):,} 個")
    
    # 2. 特徵工程
    print("\n2. 進行特徵工程...")
    feature_engineer = FeatureEngineer()
    
    # 為每個帳戶建立特徵
    account_features = feature_engineer.create_account_features(
        transactions_df, alerts_df, predict_df
    )
    
    print(f"建立特徵: {len(account_features):,} 個帳戶")
    print(f"特徵維度: {account_features.shape[1]} 個特徵")
    
    # 3. 準備訓練和測試資料
    print("\n3. 準備訓練和測試資料...")
    
    # 分離有標籤和無標籤的資料
    labeled_data = account_features[account_features['label'].notna()]
    unlabeled_data = account_features[account_features['label'].isna()]
    
    print(f"有標籤資料: {len(labeled_data):,} 筆")
    print(f"無標籤資料: {len(unlabeled_data):,} 筆")
    
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
    
    # 4. 訓練異常檢測模型
    print("\n4. 訓練異常檢測模型...")
    detector = AnomalyDetector()
    
    # 使用多種模型進行異常檢測
    models = {
        'isolation_forest': detector.train_isolation_forest(X_train),
        'random_forest': detector.train_random_forest(X_train, y_train)
    }
    
    # 5. 模型評估
    print("\n5. 評估模型效能...")
    evaluator = ModelEvaluator()
    
    results = {}
    for model_name, model in models.items():
        print(f"\n評估 {model_name}:")
        
        if model_name == 'isolation_forest':
            # Isolation Forest 預測異常分數
            val_scores = model.decision_function(X_val)
            val_predictions = model.predict(X_val)
            val_predictions = np.where(val_predictions == -1, 1, 0)  # 轉換為 0/1
        else:
            # Random Forest 預測機率
            val_scores = model.predict_proba(X_val)[:, 1]
            val_predictions = model.predict(X_val)
        
        # 計算 F1-score
        f1 = f1_score(y_val, val_predictions)
        results[model_name] = {
            'f1_score': f1,
            'predictions': val_predictions,
            'scores': val_scores
        }
        
        print(f"F1-Score: {f1:.4f}")
    
    # 6. 選擇最佳模型並進行預測
    print("\n6. 選擇最佳模型並進行預測...")
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]
    
    print(f"最佳模型: {best_model_name}")
    print(f"最佳 F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    # 對無標籤資料進行預測
    X_unlabeled = unlabeled_data[feature_cols]
    
    if best_model_name == 'isolation_forest':
        predictions = best_model.predict(X_unlabeled)
        predictions = np.where(predictions == -1, 1, 0)
        scores = best_model.decision_function(X_unlabeled)
    else:
        predictions = best_model.predict(X_unlabeled)
        scores = best_model.predict_proba(X_unlabeled)[:, 1]
    
    # 7. 建立符合 submission_template 格式的結果
    print("\n7. 建立預測結果...")
    
    # 讀取 submission_template 以確保格式一致
    template_df = pd.read_csv('submission_template.csv')
    print(f"Template 格式: {len(template_df)} 個帳戶")
    
    # 建立預測結果 DataFrame
    submission_df = pd.DataFrame({
        'acct': unlabeled_data['acct'],
        'label': predictions.astype(int)  # 確保是整數格式
    })
    
    # 檢查是否所有預測目標都在結果中
    missing_accounts = set(template_df['acct']) - set(submission_df['acct'])
    if missing_accounts:
        print(f"警告: {len(missing_accounts)} 個帳戶在預測結果中缺失")
        # 為缺失的帳戶添加預設預測 (標記為正常)
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
    
    # 顯示前幾行結果
    print(f"\n前5行預測結果:")
    print(submission_df.head())
    
    # 統計預測分佈
    label_counts = submission_df['label'].value_counts()
    print(f"\n預測分佈:")
    print(f"正常帳戶 (0): {label_counts.get(0, 0):,}")
    print(f"警示帳戶 (1): {label_counts.get(1, 0):,}")
    
    print(f"\n=== 預測完成 ===")

if __name__ == "__main__":
    create_submission()
