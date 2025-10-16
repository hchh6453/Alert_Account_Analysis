"""
改進的異常檢測模型模組
整合多種無監督學習模型，提高異常檢測效果
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnomalyDetector:
    """進階異常檢測器 - 整合多種無監督學習模型"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.model_weights = {}
    
    def train_unsupervised_models(self, X_train):
        """訓練多種無監督學習模型"""
        print("訓練多種無監督學習模型...")
        
        # 處理 NaN 值
        X_train_clean = X_train.fillna(0)
        
        # 標準化特徵
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        
        # 1. Isolation Forest
        print("  訓練 Isolation Forest...")
        isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=300
        )
        isolation_forest.fit(X_train_scaled)
        self.models['isolation_forest'] = isolation_forest
        self.model_weights['isolation_forest'] = 0.3
        
        # 2. One-Class SVM
        print("  訓練 One-Class SVM...")
        one_class_svm = OneClassSVM(
            nu=0.1,  # 異常比例
            kernel='rbf',
            gamma='scale'
        )
        one_class_svm.fit(X_train_scaled)
        self.models['one_class_svm'] = one_class_svm
        self.model_weights['one_class_svm'] = 0.25
        
        # 3. Local Outlier Factor
        print("  訓練 Local Outlier Factor...")
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        lof.fit(X_train_scaled)
        self.models['lof'] = lof
        self.model_weights['lof'] = 0.2
        
        # 4. DBSCAN (用於聚類分析)
        print("  訓練 DBSCAN...")
        dbscan = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        dbscan.fit(X_train_scaled)
        self.models['dbscan'] = dbscan
        self.model_weights['dbscan'] = 0.15
        
        # 5. Random Forest (用於特徵重要性分析)
        print("  訓練 Random Forest (特徵分析)...")
        # 創建偽標籤用於特徵重要性分析
        pseudo_labels = self.create_pseudo_labels(X_train_scaled)
        
        random_forest = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        random_forest.fit(X_train_scaled, pseudo_labels)
        self.models['random_forest'] = random_forest
        self.model_weights['random_forest'] = 0.1
        
        print("所有無監督學習模型訓練完成")
    
    def create_pseudo_labels(self, X_scaled):
        """創建偽標籤用於特徵重要性分析"""
        # 使用 Isolation Forest 創建偽標籤
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        isolation_forest.fit(X_scaled)
        predictions = isolation_forest.predict(X_scaled)
        
        # 轉換為 0/1 標籤
        pseudo_labels = np.where(predictions == -1, 1, 0)
        return pseudo_labels
    
    def predict_ensemble(self, X_test):
        """集成預測"""
        # 處理 NaN 值
        X_test_clean = X_test.fillna(0)
        
        # 標準化特徵
        X_test_scaled = self.scaler.transform(X_test_clean)
        
        # 收集各模型的預測結果
        predictions = {}
        scores = {}
        
        # 1. Isolation Forest
        iso_pred = self.models['isolation_forest'].predict(X_test_scaled)
        iso_score = self.models['isolation_forest'].decision_function(X_test_scaled)
        predictions['isolation_forest'] = np.where(iso_pred == -1, 1, 0)
        scores['isolation_forest'] = iso_score
        
        # 2. One-Class SVM
        svm_pred = self.models['one_class_svm'].predict(X_test_scaled)
        svm_score = self.models['one_class_svm'].decision_function(X_test_scaled)
        predictions['one_class_svm'] = np.where(svm_pred == -1, 1, 0)
        scores['one_class_svm'] = svm_score
        
        # 3. Local Outlier Factor
        lof_pred = self.models['lof'].predict(X_test_scaled)
        lof_score = self.models['lof'].decision_function(X_test_scaled)
        predictions['lof'] = np.where(lof_pred == -1, 1, 0)
        scores['lof'] = lof_score
        
        # 4. DBSCAN
        dbscan_pred = self.models['dbscan'].fit_predict(X_test_scaled)
        # DBSCAN: -1 表示異常點
        predictions['dbscan'] = np.where(dbscan_pred == -1, 1, 0)
        scores['dbscan'] = -np.abs(dbscan_pred)  # 使用距離作為分數
        
        # 5. Random Forest (基於偽標籤)
        rf_pred = self.models['random_forest'].predict(X_test_scaled)
        rf_score = self.models['random_forest'].predict_proba(X_test_scaled)[:, 1]
        predictions['random_forest'] = rf_pred
        scores['random_forest'] = rf_score
        
        # 加權集成預測
        ensemble_pred = np.zeros(len(X_test_scaled))
        ensemble_score = np.zeros(len(X_test_scaled))
        
        for model_name, weight in self.model_weights.items():
            ensemble_pred += weight * predictions[model_name]
            ensemble_score += weight * scores[model_name]
        
        # 轉換為二進制預測
        ensemble_pred = np.round(ensemble_pred).astype(int)
        
        return ensemble_pred, ensemble_score, predictions, scores
    
    def get_model_performance(self, X_test, y_test):
        """評估各模型性能"""
        ensemble_pred, ensemble_score, predictions, scores = self.predict_ensemble(X_test)
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        performance = {}
        
        # 評估集成模型
        performance['ensemble'] = {
            'f1_score': f1_score(y_test, ensemble_pred, average='binary'),
            'precision': precision_score(y_test, ensemble_pred, average='binary'),
            'recall': recall_score(y_test, ensemble_pred, average='binary')
        }
        
        # 評估各單一模型
        for model_name in predictions.keys():
            if len(np.unique(y_test)) > 1:  # 確保有兩個類別
                performance[model_name] = {
                    'f1_score': f1_score(y_test, predictions[model_name], average='binary'),
                    'precision': precision_score(y_test, predictions[model_name], average='binary'),
                    'recall': recall_score(y_test, predictions[model_name], average='binary')
                }
            else:
                performance[model_name] = {
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                }
        
        return performance
    
    def get_feature_importance(self):
        """獲取特徵重要性"""
        if 'random_forest' in self.models:
            return self.models['random_forest'].feature_importances_
        return None
    
    def optimize_hyperparameters(self, X_train, y_train=None):
        """超參數優化"""
        print("優化超參數...")
        
        X_train_scaled = self.scaler.fit_transform(X_train.fillna(0))
        
        # 優化 Isolation Forest
        iso_params = {
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [100, 200, 300]
        }
        
        best_iso = None
        best_score = -1
        
        for contamination in iso_params['contamination']:
            for n_estimators in iso_params['n_estimators']:
                iso = IsolationForest(
                    contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=42
                )
                iso.fit(X_train_scaled)
                
                # 使用偽標籤評估
                pseudo_labels = self.create_pseudo_labels(X_train_scaled)
                predictions = iso.predict(X_train_scaled)
                predictions = np.where(predictions == -1, 1, 0)
                
                if len(np.unique(pseudo_labels)) > 1:
                    score = f1_score(pseudo_labels, predictions, average='binary')
                    if score > best_score:
                        best_score = score
                        best_iso = iso
        
        if best_iso is not None:
            self.models['isolation_forest'] = best_iso
            print(f"最佳 Isolation Forest 參數: contamination={best_iso.contamination}, n_estimators={best_iso.n_estimators}")
    
    def predict_anomaly(self, X_test, model_type='ensemble'):
        """預測異常"""
        if model_type == 'ensemble':
            predictions, scores, _, _ = self.predict_ensemble(X_test)
            return predictions, scores
        else:
            # 單一模型預測
            X_test_clean = X_test.fillna(0)
            X_test_scaled = self.scaler.transform(X_test_clean)
            
            if model_type == 'isolation_forest':
                pred = self.models['isolation_forest'].predict(X_test_scaled)
                score = self.models['isolation_forest'].decision_function(X_test_scaled)
                pred = np.where(pred == -1, 1, 0)
            elif model_type == 'one_class_svm':
                pred = self.models['one_class_svm'].predict(X_test_scaled)
                score = self.models['one_class_svm'].decision_function(X_test_scaled)
                pred = np.where(pred == -1, 1, 0)
            elif model_type == 'lof':
                pred = self.models['lof'].predict(X_test_scaled)
                score = self.models['lof'].decision_function(X_test_scaled)
                pred = np.where(pred == -1, 1, 0)
            else:
                raise ValueError(f"不支援的模型類型: {model_type}")
            
            return pred, score
