#!/usr/bin/env python3
"""
輕量級測試腳本 - 使用10,000筆交易資料快速評估F1-score
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from anomaly_detector import AnomalyDetector

def lightweight_test():
    """輕量級測試 - 10,000筆交易資料"""
    print("=== 輕量級F1-score評估測試 ===")
    print("使用10,000筆交易資料進行快速評估")
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    # 限制交易資料量
    transactions_df = transactions_df.head(10000)
    
    print(f"交易資料 (限制): {len(transactions_df):,} 筆")
    print(f"警示帳戶: {len(alerts_df):,} 個")
    print(f"預測目標: {len(predict_df):,} 個")
    
    # 2. 取得樣本帳戶
    print("\n2. 取得樣本帳戶...")
    from_accounts = set(transactions_df['from_acct'].unique())
    to_accounts = set(transactions_df['to_acct'].unique())
    all_accounts = list(from_accounts.union(to_accounts))
    
    # 限制帳戶數量
    max_accounts = 500
    sample_accounts = all_accounts[:max_accounts]
    
    print(f"樣本帳戶數量: {len(sample_accounts)}")
    
    # 3. 特徵工程
    print("\n3. 進行特徵工程...")
    feature_engineer = FeatureEngineer()
    
    features_list = []
    for i, account in enumerate(sample_accounts):
        if i % 50 == 0:
            print(f"  處理進度: {i+1}/{len(sample_accounts)}")
        
        account_features = feature_engineer.create_single_account_features(
            account, transactions_df, alerts_df, predict_df
        )
        features_list.append(account_features)
    
    features_df = pd.DataFrame(features_list)
    
    print(f"建立特徵: {len(features_df)} 個帳戶")
    print(f"特徵維度: {features_df.shape[1]} 個特徵")
    
    # 4. 分析標籤分佈
    print("\n4. 分析標籤分佈...")
    label_counts = features_df['label'].value_counts(dropna=False)
    print(f"正常帳戶 (0): {label_counts.get(0, 0)}")
    print(f"警示帳戶 (1): {label_counts.get(1, 0)}")
    print(f"無標籤 (NaN): {features_df['label'].isna().sum()}")
    
    # 5. 準備訓練資料
    print("\n5. 準備訓練資料...")
    
    # 分離有標籤和無標籤的資料
    labeled_data = features_df[features_df['label'].notna()]
    unlabeled_data = features_df[features_df['label'].isna()]
    
    print(f"有標籤資料: {len(labeled_data)} 筆")
    print(f"無標籤資料: {len(unlabeled_data)} 筆")
    
    if len(labeled_data) < 10:
        print("警告: 有標籤資料太少，無法進行有效評估")
        print("建議增加交易資料量或調整樣本範圍")
        return
    
    # 分離特徵和標籤
    feature_cols = [col for col in features_df.columns if col not in ['acct', 'label']]
    X = labeled_data[feature_cols]
    y = labeled_data['label']
    
    # 分割訓練和驗證資料
    test_size = 0.3 if len(labeled_data) > 20 else 0.2
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"訓練資料: {len(X_train)} 筆")
    print(f"驗證資料: {len(X_val)} 筆")
    
    # 6. 訓練模型
    print("\n6. 訓練模型...")
    detector = AnomalyDetector()
    
    # 使用多種模型
    models = {}
    
    try:
        print("  訓練 Isolation Forest...")
        iso_model = detector.train_isolation_forest(X_train)
        models['isolation_forest'] = iso_model
    except Exception as e:
        print(f"  Isolation Forest 訓練失敗: {e}")
    
    try:
        print("  訓練 Random Forest...")
        rf_model = detector.train_random_forest(X_train, y_train)
        models['random_forest'] = rf_model
    except Exception as e:
        print(f"  Random Forest 訓練失敗: {e}")
    
    # 7. 模型評估
    print("\n7. 評估模型效能...")
    
    results = {}
    for model_name, model in models.items():
        print(f"\n評估 {model_name}:")
        
        try:
            if model_name == 'isolation_forest':
                # Isolation Forest 預測
                val_predictions = model.predict(X_val)
                val_predictions = np.where(val_predictions == -1, 1, 0)
                val_scores = model.decision_function(X_val)
            else:
                # Random Forest 預測
                val_predictions = model.predict(X_val)
                val_scores = model.predict_proba(X_val)[:, 1]
            
            # 計算 F1-score
            f1 = f1_score(y_val, val_predictions)
            results[model_name] = {
                'f1_score': f1,
                'predictions': val_predictions,
                'scores': val_scores
            }
            
            print(f"  F1-Score: {f1:.4f}")
            
            # 混淆矩陣
            try:
                cm = confusion_matrix(y_val, val_predictions)
                print(f"  混淆矩陣:")
                if cm.shape == (2, 2):
                    print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
                    print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
                else:
                    print(f"    混淆矩陣形狀: {cm.shape}")
                    print(f"    矩陣內容: {cm}")
            except Exception as e:
                print(f"    混淆矩陣計算失敗: {e}")
            
            # 詳細分類報告
            try:
                print(f"  分類報告:")
                report = classification_report(y_val, val_predictions, target_names=['正常', '警示'])
                print(report)
            except Exception as e:
                print(f"    分類報告生成失敗: {e}")
            
        except Exception as e:
            print(f"  評估失敗: {e}")
    
    # 8. 結果總結
    print("\n8. 結果總結...")
    
    if results:
        best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_f1 = results[best_model]['f1_score']
        
        print(f"最佳模型: {best_model}")
        print(f"最佳 F1-Score: {best_f1:.4f}")
        
        print(f"\n所有模型 F1-Score:")
        for model_name, result in results.items():
            print(f"  {model_name}: {result['f1_score']:.4f}")
    else:
        print("沒有成功訓練的模型")
    
    # 9. 預測示例
    if len(unlabeled_data) > 0 and results:
        print(f"\n9. 預測示例...")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_model = models[best_model_name]
        
        # 對無標籤資料進行預測
        X_unlabeled = unlabeled_data[feature_cols]
        
        if best_model_name == 'isolation_forest':
            predictions = best_model.predict(X_unlabeled)
            predictions = np.where(predictions == -1, 1, 0)
        else:
            predictions = best_model.predict(X_unlabeled)
        
        print(f"預測警示帳戶數量: {predictions.sum()}")
        print(f"預測正常帳戶數量: {(predictions == 0).sum()}")
        
        # 儲存預測結果
        result_df = pd.DataFrame({
            'acct': unlabeled_data['acct'],
            'label': predictions.astype(int)
        })
        result_df.to_csv('lightweight_prediction.csv', index=False)
        print(f"預測結果已儲存至 lightweight_prediction.csv")
    
    print(f"\n=== 輕量級測試完成 ===")
    print(f"使用資料: {len(transactions_df):,} 筆交易")
    print(f"處理帳戶: {len(sample_accounts)} 個")
    print(f"評估模型: {len(models)} 個")

if __name__ == "__main__":
    lightweight_test()
