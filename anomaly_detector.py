"""
異常檢測模型模組
實作多種異常檢測演算法，包括 Isolation Forest 和 Random Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """異常檢測器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
    
    def train_isolation_forest(self, X_train):
        """訓練 Isolation Forest 模型"""
        print("訓練 Isolation Forest 模型...")
        
        # 處理 NaN 值
        X_train_clean = X_train.fillna(0)  # 用0填充NaN值
        
        # 標準化特徵
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        
        # 設定參數 - 調整異常比例
        params = {
            'contamination': 0.1,  # 調整到 10%
            'random_state': 42,
            'n_estimators': 300    # 增加樹的數量
        }
        
        # 訓練模型
        model = IsolationForest(**params)
        model.fit(X_train_scaled)
        
        print("Isolation Forest 訓練完成")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """訓練 Random Forest 模型"""
        print("訓練 Random Forest 模型...")
        
        # 處理 NaN 值
        X_train_clean = X_train.fillna(0)  # 用0填充NaN值
        
        # 標準化特徵
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        
        # 設定參數
        params = {
            'n_estimators': 300,    # 增加樹的數量
            'max_depth': 20,        # 增加深度
            'min_samples_split': 2, # 減少分割要求
            'min_samples_leaf': 1,  # 減少葉節點要求
            'random_state': 42,
            'class_weight': 'balanced'  # 處理不平衡資料
        }
        
        # 訓練模型
        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)
        
        print("Random Forest 訓練完成")
        return model
    
    def train_ensemble_model(self, X_train, y_train):
        """訓練集成模型"""
        print("訓練集成模型...")
        
        # 訓練多個基學習器
        models = {
            'isolation_forest': self.train_isolation_forest(X_train),
            'random_forest': self.train_random_forest(X_train, y_train)
        }
        
        return models
    
    def predict_anomaly(self, model, X_test, model_type='isolation_forest'):
        """預測異常"""
        # 處理 NaN 值
        X_test_clean = X_test.fillna(0)  # 用0填充NaN值
        
        # 標準化特徵
        X_test_scaled = self.scaler.transform(X_test_clean)
        
        if model_type == 'isolation_forest':
            # Isolation Forest 預測
            predictions = model.predict(X_test_scaled)
            scores = model.decision_function(X_test_scaled)
            
            # 轉換為 0/1 標籤 (-1 -> 1, 1 -> 0)
            predictions = np.where(predictions == -1, 1, 0)
            
        else:
            # Random Forest 預測
            predictions = model.predict(X_test_scaled)
            scores = model.predict_proba(X_test_scaled)[:, 1]
        
        return predictions, scores
    
    def get_feature_importance(self, model, feature_names):
        """取得特徵重要性"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='random_forest'):
        """超參數優化"""
        print(f"優化 {model_type} 超參數...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if model_type == 'random_forest':
            # Random Forest 參數網格
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
            
        elif model_type == 'isolation_forest':
            # Isolation Forest 參數網格
            param_grid = {
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'n_estimators': [50, 100, 200]
            }
            
            model = IsolationForest(random_state=42)
        
        # 網格搜索
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"最佳參數: {grid_search.best_params_}")
        print(f"最佳分數: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def create_ensemble_prediction(self, models, X_test):
        """建立集成預測"""
        predictions = []
        scores = []
        
        for model_name, model in models.items():
            pred, score = self.predict_anomaly(model, X_test, model_name)
            predictions.append(pred)
            scores.append(score)
        
        # 簡單投票
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        
        # 平均分數
        ensemble_score = np.mean(scores, axis=0)
        
        return ensemble_pred, ensemble_score
