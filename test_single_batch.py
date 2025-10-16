"""
測試單一批次的異常檢測效果
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 匯入必要的模組
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
except ImportError as e:
    print(f"匯入錯誤: {e}")
    print("請執行: pip install scikit-learn")
    exit(1)

from data_loader import DataLoader
from alert_focused_feature_engineering import AlertFocusedFeatureEngineer
from advanced_anomaly_detector import AdvancedAnomalyDetector

def test_single_batch():
    """測試單一批次"""
    print("=== 測試單一批次異常檢測 ===")
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    print(f"原始交易資料: {len(transactions_df):,} 筆")
    print(f"警示帳戶: {len(alerts_df):,} 個")
    print(f"預測目標: {len(predict_df):,} 個")
    
    # 2. 選擇第一個批次
    batch_size = 100000  # 10萬筆交易
    batch_transactions = transactions_df.head(batch_size)
    print(f"\n批次交易資料: {len(batch_transactions):,} 筆")
    
    # 3. 取得批次中的帳戶
    batch_from_accounts = set(batch_transactions['from_acct'].unique())
    batch_to_accounts = set(batch_transactions['to_acct'].unique())
    batch_accounts = list(batch_from_accounts.union(batch_to_accounts))
    
    print(f"批次帳戶數量: {len(batch_accounts)}")
    
    # 4. 檢查警示帳戶
    alert_accounts = set(alerts_df['acct'])
    batch_alert_accounts = alert_accounts.intersection(set(batch_accounts))
    print(f"批次警示帳戶: {len(batch_alert_accounts)}")
    
    if len(batch_alert_accounts) == 0:
        print("❌ 批次中無警示帳戶，無法測試")
        return
    
    # 5. 限制帳戶數量
    max_accounts = 3000
    if len(batch_accounts) > max_accounts:
        # 優先保留警示帳戶
        remaining_slots = max_accounts - len(batch_alert_accounts)
        if remaining_slots > 0:
            other_accounts = [acc for acc in batch_accounts if acc not in batch_alert_accounts]
            import random
            random.seed(42)
            selected_other = random.sample(other_accounts, min(remaining_slots, len(other_accounts)))
            batch_accounts = list(batch_alert_accounts) + selected_other
        else:
            batch_accounts = list(batch_alert_accounts)
    
    print(f"限制後批次帳戶: {len(batch_accounts)}")
    
    # 6. 建立特徵
    print("\n2. 建立特徵...")
    feature_engineer = AlertFocusedFeatureEngineer()
    
    batch_features = []
    for i, account in enumerate(batch_accounts):
        if i % 100 == 0:
            print(f"  處理進度: {i+1}/{len(batch_accounts)}")
        
        account_features = feature_engineer.create_alert_focused_features(
            account, batch_transactions, alerts_df, predict_df
        )
        batch_features.append(account_features)
    
    batch_features_df = pd.DataFrame(batch_features)
    print(f"批次特徵: {len(batch_features_df):,} 個帳戶, {batch_features_df.shape[1]} 個特徵")
    
    # 7. 分離有標籤和無標籤的資料
    labeled_data = batch_features_df[batch_features_df['label'] != -1]
    unlabeled_data = batch_features_df[batch_features_df['label'] == -1]
    
    print(f"\n有標籤資料: {len(labeled_data):,} 筆")
    print(f"無標籤資料: {len(unlabeled_data):,} 筆")
    
    # 檢查標籤分布
    print(f"標籤分布: {labeled_data['label'].value_counts().to_dict()}")
    
    # 8. 分離特徵和標籤
    feature_cols = [col for col in batch_features_df.columns if col not in ['acct', 'label']]
    X = labeled_data[feature_cols]
    y = labeled_data['label']
    
    # 分割訓練和驗證資料
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練資料: {len(X_train):,} 筆")
    print(f"驗證資料: {len(X_val):,} 筆")
    
    # 9. 訓練模型
    print("\n3. 訓練模型...")
    detector = AdvancedAnomalyDetector()
    
    # 訓練多種無監督學習模型
    detector.train_unsupervised_models(X_train)
    
    # 優化超參數
    detector.optimize_hyperparameters(X_train)
    
    # 10. 評估模型
    print("\n4. 評估模型效能...")
    performance = detector.get_model_performance(X_val, y_val)
    
    print("\n=== 模型性能比較 ===")
    for model_name, metrics in performance.items():
        print(f"\n{model_name}:")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    # 選擇最佳模型
    best_model_name = max(performance.keys(), key=lambda x: performance[x]['f1_score'])
    best_f1_score = performance[best_model_name]['f1_score']
    
    print(f"\n最佳模型: {best_model_name}")
    print(f"最佳 F1-Score: {best_f1_score:.4f}")
    
    # 11. 對無標籤資料進行預測
    if len(unlabeled_data) > 0:
        print("\n5. 對無標籤資料進行預測...")
        X_unlabeled = unlabeled_data[feature_cols]
        
        predictions, scores, individual_predictions, individual_scores = detector.predict_ensemble(X_unlabeled)
        
        print(f"預測警示帳戶數量: {predictions.sum():,}")
        print(f"預測正常帳戶數量: {(predictions == 0).sum():,}")
        
        # 顯示各模型的預測結果
        print(f"\n各模型預測結果:")
        for model_name, pred in individual_predictions.items():
            alert_count = pred.sum()
            print(f"  {model_name}: {alert_count:,} 個警示帳戶")
    else:
        print("\n5. 無標籤資料為空，跳過預測")
    
    print("\n=== 單一批次測試完成 ===")
    return performance

if __name__ == "__main__":
    test_single_batch()
