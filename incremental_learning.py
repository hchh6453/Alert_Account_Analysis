"""
增量學習異常檢測系統
1. 使用已知標籤資料訓練多個模型
2. 透過交叉驗證比較模型準確率
3. 選擇最佳模型預測未知標籤資料
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
from datetime import datetime

# 匯入必要的模組
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    print(f"匯入錯誤: {e}")
    print("請執行: pip install scikit-learn")
    exit(1)

from data_loader import DataLoader
from alert_focused_feature_engineering import AlertFocusedFeatureEngineer

class IncrementalDetector:
    """增量學習異常檢測器"""
    
    def __init__(self):
        self.feature_engineer = AlertFocusedFeatureEngineer()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model_name = None
        self.best_f1_score = 0.0
        
        print(f"增量學習檢測器初始化完成")
    
    def create_training_data(self, transactions_df, alerts_df, predict_df, 
                           alert_ratio=0.3, sample_size=5000, strategy='balanced'):
        """創建訓練資料集"""
        print(f"\n=== 創建訓練資料集 ===")
        
        # 取得已知警示帳戶和正常帳戶
        alert_accounts = set(alerts_df['acct'])
        
        # 從交易資料中取得所有帳戶（from_acct 和 to_acct）
        all_from_accounts = set(transactions_df['from_acct'].unique())
        all_to_accounts = set(transactions_df['to_acct'].unique())
        all_accounts_in_txns = all_from_accounts.union(all_to_accounts)
        
        predict_accounts = set(predict_df['acct'])
        normal_accounts = all_accounts_in_txns - predict_accounts - alert_accounts
        
        print(f"已知警示帳戶: {len(alert_accounts)}")
        print(f"可用正常帳戶: {len(normal_accounts)}")
        
        # 計算需要的帳戶數量
        target_alert_count = int(sample_size * alert_ratio)
        target_normal_count = sample_size - target_alert_count
        
        # 採樣警示帳戶
        if len(alert_accounts) >= target_alert_count:
            sampled_alerts = list(alert_accounts)[:target_alert_count]
        else:
            sampled_alerts = list(alert_accounts) * (target_alert_count // len(alert_accounts) + 1)
            sampled_alerts = sampled_alerts[:target_alert_count]
        
        # 採樣正常帳戶（根據策略）
        if strategy == 'balanced':
            # 策略1：平衡採樣（簡單隨機）- 快速但可能偏向
            if len(normal_accounts) >= target_normal_count:
                sampled_normal = list(normal_accounts)[:target_normal_count]
            else:
                sampled_normal = list(normal_accounts)
                
        elif strategy == 'diverse':
            # 策略2：多樣化採樣（確保不同類型的正常帳戶）- 隨機性強
            normal_list = list(normal_accounts)
            if len(normal_list) >= target_normal_count:
                # 隨機採樣確保多樣性
                import random
                random.seed(42)
                sampled_normal = random.sample(normal_list, target_normal_count)
            else:
                sampled_normal = normal_list
                
        elif strategy == 'active':
            # 策略3：活躍帳戶優先（交易次數多的帳戶）- 重視交易頻繁的帳戶
            # 計算每個正常帳戶的交易次數
            normal_txn_counts = {}
            for account in list(normal_accounts)[:10000]:  # 限制計算範圍避免太慢
                account_txns = self.feature_engineer.get_account_transactions(account, transactions_df)
                normal_txn_counts[account] = len(account_txns)
            
            # 按交易次數排序，選擇最活躍的帳戶
            sorted_accounts = sorted(normal_txn_counts.items(), key=lambda x: x[1], reverse=True)
            sampled_normal = [account for account, count in sorted_accounts[:target_normal_count]]
            
        elif strategy == 'representative':
            # 策略4：代表性採樣（確保涵蓋不同交易模式）
            # 先隨機選擇一部分帳戶，分析其交易特徵
            import random
            random.seed(42)
            sample_size_for_analysis = min(5000, len(normal_accounts))
            analysis_accounts = random.sample(list(normal_accounts), sample_size_for_analysis)
            
            # 計算這些帳戶的交易特徵
            account_features = []
            for account in analysis_accounts:
                account_txns = self.feature_engineer.get_account_transactions(account, transactions_df)
                if len(account_txns) > 0:
                    avg_amount = account_txns['txn_amt'].mean()
                    txn_count = len(account_txns)
                    account_features.append((account, avg_amount, txn_count))
            
            # 按特徵分組，從每組中選擇代表性帳戶
            account_features.sort(key=lambda x: x[1])  # 按平均金額排序
            group_size = len(account_features) // 10  # 分成10組
            sampled_normal = []
            
            for i in range(0, len(account_features), group_size):
                group = account_features[i:i+group_size]
                if group:
                    # 從每組中隨機選擇
                    selected = random.choice(group)
                    sampled_normal.append(selected[0])
                    if len(sampled_normal) >= target_normal_count:
                        break
            
            # 如果還不夠，隨機補充
            if len(sampled_normal) < target_normal_count:
                remaining_needed = target_normal_count - len(sampled_normal)
                remaining_accounts = list(normal_accounts - set(sampled_normal))
                additional = random.sample(remaining_accounts, min(remaining_needed, len(remaining_accounts)))
                sampled_normal.extend(additional)
        else:
            # 預設策略
            if len(normal_accounts) >= target_normal_count:
                sampled_normal = list(normal_accounts)[:target_normal_count]
            else:
                sampled_normal = list(normal_accounts)
        
        print(f"採樣警示帳戶: {len(sampled_alerts)}")
        print(f"採樣正常帳戶: {len(sampled_normal)}")
        
        # 建立特徵
        training_features = []
        
        print("建立警示帳戶特徵...")
        for i, account in enumerate(sampled_alerts):
            if i % 100 == 0:
                print(f"  進度: {i+1}/{len(sampled_alerts)}")
            
            features = self.feature_engineer.create_alert_focused_features(
                account, transactions_df, alerts_df, predict_df
            )
            training_features.append(features)
        
        print("建立正常帳戶特徵...")
        for i, account in enumerate(sampled_normal):
            if i % 100 == 0:
                print(f"  進度: {i+1}/{len(sampled_normal)}")
            
            features = self.feature_engineer.create_alert_focused_features(
                account, transactions_df, alerts_df, predict_df
            )
            training_features.append(features)
        
        # 轉換為 DataFrame
        training_df = pd.DataFrame(training_features)
        feature_cols = [col for col in training_df.columns if col not in ['acct', 'label']]
        X_train = training_df[feature_cols]
        y_train = training_df['label']
        
        print(f"訓練資料形狀: {X_train.shape}")
        print(f"標籤分布: {y_train.value_counts().to_dict()}")
        
        return X_train, y_train, feature_cols
    
    def train_and_compare_models(self, X_train, y_train):
        """訓練並比較多個模型"""
        print(f"\n=== 訓練並比較模型 ===")
        
        # 處理資料
        X_train_clean = X_train.fillna(0)
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        
        # 分割訓練和驗證資料
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"訓練資料: {len(X_train_split)} 筆")
        print(f"驗證資料: {len(X_val_split)} 筆")
        
        # 使用交叉驗證評估模型
        print(f"\n=== 使用交叉驗證評估模型 ===")
        cv_scores = {}
        
        # 1. Random Forest
        print("1. 評估 Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'  # 重新啟用類別權重
        )
        cv_f1_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='f1')
        cv_scores['random_forest'] = cv_f1_scores.mean()
        print(f"  交叉驗證 F1-Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std() * 2:.4f})")
        
        rf.fit(X_train_split, y_train_split)
        self.models['random_forest'] = rf
        
        # 2. Isolation Forest
        print("2. 評估 Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=0.1,  # 10% 異常比例（平衡學習）
            random_state=42,
            n_estimators=100
        )
        iso_predictions = iso_forest.fit_predict(X_train_scaled)
        iso_predictions = np.where(iso_predictions == -1, 1, 0)
        iso_f1 = f1_score(y_train, iso_predictions, average='binary')
        cv_scores['isolation_forest'] = iso_f1
        print(f"  訓練資料 F1-Score: {iso_f1:.4f}")
        
        iso_forest.fit(X_train_split)
        self.models['isolation_forest'] = iso_forest
        
        # 3. One-Class SVM
        print("3. 評估 One-Class SVM...")
        oc_svm = OneClassSVM(
            nu=0.1,  # 10% 異常比例（平衡學習）
            kernel='rbf',
            gamma='scale'
        )
        oc_svm.fit(X_train_scaled)
        oc_predictions = oc_svm.predict(X_train_scaled)
        oc_predictions = np.where(oc_predictions == -1, 1, 0)
        oc_f1 = f1_score(y_train, oc_predictions, average='binary')
        cv_scores['one_class_svm'] = oc_f1
        print(f"  訓練資料 F1-Score: {oc_f1:.4f}")
        
        self.models['one_class_svm'] = oc_svm
        
        # 4. Local Outlier Factor
        print("4. 評估 Local Outlier Factor...")
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,  # 10% 異常比例（平衡學習）
            novelty=True
        )
        lof.fit(X_train_scaled)
        lof_predictions = lof.predict(X_train_scaled)
        lof_predictions = np.where(lof_predictions == -1, 1, 0)
        lof_f1 = f1_score(y_train, lof_predictions, average='binary')
        cv_scores['lof'] = lof_f1
        print(f"  訓練資料 F1-Score: {lof_f1:.4f}")
        
        self.models['lof'] = lof
        
        # 5. DBSCAN
        print("5. 評估 DBSCAN...")
        dbscan = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        dbscan.fit(X_train_scaled)
        dbscan_predictions = dbscan.fit_predict(X_train_scaled)
        dbscan_predictions = np.where(dbscan_predictions == -1, 1, 0)
        dbscan_f1 = f1_score(y_train, dbscan_predictions, average='binary')
        cv_scores['dbscan'] = dbscan_f1
        print(f"  訓練資料 F1-Score: {dbscan_f1:.4f}")
        
        self.models['dbscan'] = dbscan
        
        # 選擇最佳模型
        best_model_name = max(cv_scores.keys(), key=lambda x: cv_scores[x])
        best_f1_score = cv_scores[best_model_name]
        
        self.best_model_name = best_model_name
        self.best_f1_score = best_f1_score
        
        print(f"\n🎯 最佳模型: {best_model_name}")
        print(f"🎯 最佳 F1-Score: {best_f1_score:.4f}")
        
        # 顯示所有模型排名
        print(f"\n📊 模型排名 (按 F1-Score):")
        sorted_models = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, score) in enumerate(sorted_models, 1):
            print(f"  {i}. {model_name}: {score:.4f}")
        
        # 在驗證資料上評估最佳模型
        print(f"\n=== 驗證最佳模型 {best_model_name} ===")
        self.evaluate_best_model(X_val_split, y_val_split)
    
    def evaluate_best_model(self, X_val, y_val):
        """評估最佳模型"""
        best_model = self.models[self.best_model_name]
        
        try:
            if self.best_model_name == 'random_forest':
                predictions = best_model.predict(X_val)
                scores = best_model.predict_proba(X_val)[:, 1]
            elif self.best_model_name == 'isolation_forest':
                predictions = best_model.predict(X_val)
                predictions = np.where(predictions == -1, 1, 0)
                scores = best_model.decision_function(X_val)
            elif self.best_model_name == 'one_class_svm':
                predictions = best_model.predict(X_val)
                predictions = np.where(predictions == -1, 1, 0)
                scores = best_model.decision_function(X_val)
            elif self.best_model_name == 'lof':
                predictions = best_model.predict(X_val)
                predictions = np.where(predictions == -1, 1, 0)
                scores = best_model.decision_function(X_val)
            elif self.best_model_name == 'dbscan':
                predictions = best_model.fit_predict(X_val)
                predictions = np.where(predictions == -1, 1, 0)
                scores = -np.abs(predictions)
            
            # 計算性能指標
            if len(np.unique(y_val)) > 1:
                f1 = f1_score(y_val, predictions, average='binary')
                precision = precision_score(y_val, predictions, average='binary')
                recall = recall_score(y_val, predictions, average='binary')
            else:
                f1 = precision = recall = 0.0
            
            print(f"驗證結果:")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            
            # 顯示混淆矩陣
            cm = confusion_matrix(y_val, predictions)
            print(f"  混淆矩陣: {cm}")
            
        except Exception as e:
            print(f"❌ 評估最佳模型失敗: {e}")
    
    def predict_with_best_model(self, X_test, feature_cols):
        """使用最佳模型進行預測"""
        if self.best_model_name is None:
            print("❌ 沒有可用的最佳模型")
            return None
        
        print(f"🎯 使用最佳模型 {self.best_model_name} 進行預測...")
        
        X_test_clean = X_test.fillna(0)
        X_test_scaled = self.scaler.transform(X_test_clean)
        
        best_model = self.models[self.best_model_name]
        
        if self.best_model_name == 'random_forest':
            predictions = best_model.predict(X_test_scaled)
            scores = best_model.predict_proba(X_test_scaled)[:, 1]
        elif self.best_model_name == 'isolation_forest':
            predictions = best_model.predict(X_test_scaled)
            predictions = np.where(predictions == -1, 1, 0)
            scores = best_model.decision_function(X_test_scaled)
        elif self.best_model_name == 'one_class_svm':
            predictions = best_model.predict(X_test_scaled)
            predictions = np.where(predictions == -1, 1, 0)
            scores = best_model.decision_function(X_test_scaled)
        elif self.best_model_name == 'lof':
            predictions = best_model.predict(X_test_scaled)
            predictions = np.where(predictions == -1, 1, 0)
            scores = best_model.decision_function(X_test_scaled)
        elif self.best_model_name == 'dbscan':
            predictions = best_model.fit_predict(X_test_scaled)
            predictions = np.where(predictions == -1, 1, 0)
            scores = -np.abs(predictions)
        
        return predictions
    
    def predict_all_accounts(self, transactions_df, alerts_df, predict_df, feature_cols):
        """預測所有需要預測的帳戶"""
        print(f"\n=== 預測所有帳戶 ===")
        
        predict_accounts = list(predict_df['acct'])
        batch_size = 1000
        
        all_predictions = []
        all_accounts = []
        
        for batch_num in range(0, len(predict_accounts), batch_size):
            batch_accounts = predict_accounts[batch_num:batch_num + batch_size]
            print(f"\n處理批次 {batch_num // batch_size + 1}: {len(batch_accounts)} 個帳戶")
            
            # 建立特徵
            batch_features = []
            for i, account in enumerate(batch_accounts):
                if i % 100 == 0:
                    print(f"  進度: {i+1}/{len(batch_accounts)}")
                
                features = self.feature_engineer.create_alert_focused_features(
                    account, transactions_df, alerts_df, predict_df
                )
                batch_features.append(features)
            
            batch_df = pd.DataFrame(batch_features)
            X_batch = batch_df[feature_cols]
            
            # 使用最佳模型預測
            predictions = self.predict_with_best_model(X_batch, feature_cols)
            
            if predictions is not None:
                all_predictions.extend(predictions)
                all_accounts.extend(batch_accounts)
                
                print(f"預測結果: {predictions.sum()} 個警示帳戶")
        
        return all_predictions, all_accounts

def run_incremental_learning():
    """運行增量學習"""
    print("=== 增量學習異常檢測系統 ===")
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    print(f"原始交易資料: {len(transactions_df):,} 筆")
    print(f"警示帳戶: {len(alerts_df):,} 個")
    print(f"預測目標: {len(predict_df):,} 個")
    
    # 2. 初始化檢測器
    print("\n2. 初始化檢測器...")
    detector = IncrementalDetector()
    
    # 3. 創建訓練資料
    X_train, y_train, feature_cols = detector.create_training_data(
        transactions_df, alerts_df, predict_df,
        alert_ratio=0.01,  # 1% 警示帳戶（平衡學習和真實性）
        sample_size=5000,  # 5000 個訓練樣本（減少計算時間）
        strategy='representative'  # 追求全面性：確保涵蓋不同交易模式
        # 其他策略選項（已註解保留）:
        # strategy='balanced'   # 平衡採樣：簡單隨機選擇
        # strategy='diverse'    # 多樣化採樣：隨機採樣確保多樣性
        # strategy='active'     # 活躍帳戶優先：選擇交易次數多的帳戶
    )
    
    # 4. 訓練並比較模型
    detector.train_and_compare_models(X_train, y_train)
    
    # 5. 使用最佳模型預測所有帳戶
    all_predictions, all_accounts = detector.predict_all_accounts(
        transactions_df, alerts_df, predict_df, feature_cols
    )
    
    # 6. 輸出結果
    if all_predictions:
        print(f"\n=== 最終結果 ===")
        
        # 建立符合 submission_template 格式的結果
        submission_df = pd.DataFrame({
            'acct': all_accounts,
            'label': all_predictions
        })
        
        # 儲存結果
        submission_df.to_csv('submission.csv', index=False)
        
        print(f"預測結果已儲存至 submission.csv")
        print(f"總預測警示帳戶: {sum(all_predictions):,}")
        print(f"總預測正常帳戶: {len(all_predictions) - sum(all_predictions):,}")
        
        # 顯示最佳模型信息
        print(f"\n最終最佳模型: {detector.best_model_name}")
        print(f"最終最佳 F1-Score: {detector.best_f1_score:.4f}")
        
        # 驗證完整性
        predict_accounts = set(predict_df['acct'])
        submission_accounts = set(submission_df['acct'])
        
        if predict_accounts == submission_accounts:
            print("✅ 預測完整性驗證通過")
        else:
            print("❌ 預測完整性驗證失敗")
            print(f"缺少: {len(predict_accounts - submission_accounts)}")
            print(f"多餘: {len(submission_accounts - predict_accounts)}")
    
    return detector

if __name__ == "__main__":
    run_incremental_learning()