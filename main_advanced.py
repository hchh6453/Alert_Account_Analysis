"""
金融交易異常檢測系統 - 主程式
使用進階無監督學習模型進行異常檢測
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

def main():
    """主程式流程"""
    print("=== 金融交易異常檢測系統 (進階版) ===")
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    # 限制交易資料量以避免電力耗盡
    original_size = len(transactions_df)
    transactions_df = transactions_df.head(500000)  # 50 萬筆
    print(f"原始交易資料: {original_size:,} 筆")
    print(f"限制後交易資料: {len(transactions_df):,} 筆")
    print(f"警示帳戶: {len(alerts_df):,} 個")
    print(f"預測目標: {len(predict_df):,} 個")
    
    # 2. 特徵工程
    print("\n2. 進行警示導向的特徵工程...")
    feature_engineer = AlertFocusedFeatureEngineer()
    
    # 限制帳戶數量以避免處理時間過長
    from_accounts = set(transactions_df['from_acct'].unique())
    to_accounts = set(transactions_df['to_acct'].unique())
    all_accounts = list(from_accounts.union(to_accounts))
    
    # 限制為前 10000 個帳戶，保持真實分布
    max_accounts = 10000
    if len(all_accounts) > max_accounts:
        print(f"限制帳戶數量: {len(all_accounts)} -> {max_accounts}")
        
        # 隨機取樣，保持真實的警示帳戶比例
        import random
        random.seed(42)
        sample_accounts = random.sample(all_accounts, max_accounts)
        
        # 檢查取樣後的警示帳戶比例
        alert_accounts = set(alerts_df['acct'])
        alert_accounts_in_sample = alert_accounts.intersection(set(sample_accounts))
        
        print(f"取樣後警示帳戶: {len(alert_accounts_in_sample)} 個")
        print(f"取樣後警示帳戶比例: {len(alert_accounts_in_sample)/max_accounts:.4f}")
        
        # 只為樣本帳戶建立特徵
        features_list = []
        for i, account in enumerate(sample_accounts):
            if i % 100 == 0:
                print(f"  處理進度: {i+1}/{len(sample_accounts)}")
            
            account_features = feature_engineer.create_alert_focused_features(
                account, transactions_df, alerts_df, predict_df
            )
            features_list.append(account_features)
        
        account_features = pd.DataFrame(features_list)
    else:
        # 為每個帳戶建立特徵
        features_list = []
        for i, account in enumerate(all_accounts):
            if i % 100 == 0:
                print(f"  處理進度: {i+1}/{len(all_accounts)}")
            
            account_features = feature_engineer.create_alert_focused_features(
                account, transactions_df, alerts_df, predict_df
            )
            features_list.append(account_features)
        
        account_features = pd.DataFrame(features_list)
    
    print(f"建立特徵: {len(account_features):,} 個帳戶")
    print(f"特徵維度: {account_features.shape[1]} 個特徵")
    
    # 3. 準備訓練和測試資料
    print("\n3. 準備訓練和測試資料...")
    
    # 分離有標籤和無標籤的資料
    labeled_data = account_features[account_features['label'] != -1]
    unlabeled_data = account_features[account_features['label'] == -1]
    
    print(f"有標籤資料: {len(labeled_data):,} 筆")
    print(f"無標籤資料: {len(unlabeled_data):,} 筆")
    
    # 檢查標籤分布
    print(f"標籤分布: {labeled_data['label'].value_counts().to_dict()}")
    
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
    
    # 4. 訓練進階異常檢測模型
    print("\n4. 訓練進階異常檢測模型...")
    detector = AdvancedAnomalyDetector()
    
    # 訓練多種無監督學習模型
    detector.train_unsupervised_models(X_train)
    
    # 優化超參數
    detector.optimize_hyperparameters(X_train)
    
    # 5. 模型評估
    print("\n5. 評估模型效能...")
    
    # 評估集成模型和各單一模型
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
    
    # 6. 選擇最佳模型並進行預測
    print("\n6. 選擇最佳模型並進行預測...")
    
    # 對無標籤資料進行預測
    X_unlabeled = unlabeled_data[feature_cols]
    
    # 使用集成模型進行預測
    predictions, scores, individual_predictions, individual_scores = detector.predict_ensemble(X_unlabeled)
    
    # 7. 輸出結果
    print("\n7. 輸出預測結果...")
    
    # 建立結果 DataFrame (符合 submission_template 格式)
    result_df = pd.DataFrame({
        'acct': unlabeled_data['acct'],
        'label': predictions  # 使用 'label' 欄位名稱，符合 template 格式
    })
    
    # 儲存結果 (符合 submission 格式)
    result_df.to_csv('submission.csv', index=False)
    
    print(f"預測結果已儲存至 submission.csv")
    print(f"預測警示帳戶數量: {predictions.sum():,}")
    print(f"預測正常帳戶數量: {(predictions == 0).sum():,}")
    
    # 同時儲存詳細結果 (包含分數)
    detailed_result_df = pd.DataFrame({
        'acct': unlabeled_data['acct'],
        'label': predictions,
        'score': scores
    })
    detailed_result_df.to_csv('prediction_results.csv', index=False)
    print(f"詳細結果已儲存至 prediction_results.csv")
    
    # 8. 分析結果
    print("\n8. 結果分析...")
    
    # 顯示高風險帳戶
    high_risk_accounts = detailed_result_df[detailed_result_df['label'] == 1].sort_values('score', ascending=False)
    print(f"\n前10個高風險帳戶:")
    print(high_risk_accounts.head(10)[['acct', 'score']])
    
    # 顯示各模型的預測結果
    print(f"\n各模型預測結果:")
    for model_name, pred in individual_predictions.items():
        alert_count = pred.sum()
        print(f"  {model_name}: {alert_count:,} 個警示帳戶")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()
