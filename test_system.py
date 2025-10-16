#!/usr/bin/env python3
"""
測試腳本
用於測試系統各個模組是否正常運作
"""

import sys
import traceback

def test_imports():
    """測試模組匯入"""
    print("測試模組匯入...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import IsolationForest, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import f1_score
        print("✓ 基礎套件匯入成功")
    except ImportError as e:
        print(f"✗ 基礎套件匯入失敗: {e}")
        return False
    
    try:
        from data_loader import DataLoader
        print("✓ DataLoader 匯入成功")
    except ImportError as e:
        print(f"✗ DataLoader 匯入失敗: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✓ FeatureEngineer 匯入成功")
    except ImportError as e:
        print(f"✗ FeatureEngineer 匯入失敗: {e}")
        return False
    
    try:
        from anomaly_detector import AnomalyDetector
        print("✓ AnomalyDetector 匯入成功")
    except ImportError as e:
        print(f"✗ AnomalyDetector 匯入失敗: {e}")
        return False
    
    try:
        from evaluator import ModelEvaluator
        print("✓ ModelEvaluator 匯入成功")
    except ImportError as e:
        print(f"✗ ModelEvaluator 匯入失敗: {e}")
        return False
    
    return True

def test_data_loading():
    """測試資料載入"""
    print("\n測試資料載入...")
    
    try:
        from data_loader import DataLoader
        loader = DataLoader()
        
        # 檢查資料檔案是否存在
        import os
        data_files = [
            "data/acct_transaction.csv",
            "data/acct_alert.csv", 
            "data/acct_predict.csv"
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                print(f"✓ {file_path} 存在")
            else:
                print(f"✗ {file_path} 不存在")
                return False
        
        print("✓ 所有資料檔案都存在")
        return True
        
    except Exception as e:
        print(f"✗ 資料載入測試失敗: {e}")
        traceback.print_exc()
        return False

def test_feature_engineering():
    """測試特徵工程"""
    print("\n測試特徵工程...")
    
    try:
        from feature_engineering import FeatureEngineer
        import pandas as pd
        import numpy as np
        
        # 建立測試資料
        test_transactions = pd.DataFrame({
            'from_acct': ['A', 'B', 'A', 'C'],
            'to_acct': ['B', 'A', 'C', 'A'],
            'txn_amt': [1000, 2000, 1500, 3000],
            'txn_date': [1, 2, 3, 4],
            'txn_time': ['10:00:00', '11:00:00', '12:00:00', '13:00:00'],
            'is_self_txn': ['N', 'N', 'N', 'N']
        })
        
        test_alerts = pd.DataFrame({
            'acct': ['A'],
            'event_date': [1]
        })
        
        test_predict = pd.DataFrame({
            'acct': ['B', 'C'],
            'label': [0, 1]
        })
        
        # 測試特徵工程
        engineer = FeatureEngineer()
        features = engineer.create_account_features(
            test_transactions, test_alerts, test_predict
        )
        
        print(f"✓ 特徵工程成功，產生 {len(features)} 個帳戶特徵")
        print(f"✓ 特徵維度: {features.shape[1]} 個特徵")
        
        return True
        
    except Exception as e:
        print(f"✗ 特徵工程測試失敗: {e}")
        traceback.print_exc()
        return False

def test_anomaly_detection():
    """測試異常檢測"""
    print("\n測試異常檢測...")
    
    try:
        from anomaly_detector import AnomalyDetector
        import numpy as np
        
        # 建立測試資料
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        # 測試異常檢測器
        detector = AnomalyDetector()
        
        # 測試 Isolation Forest
        iso_model = detector.train_isolation_forest(X_train)
        print("✓ Isolation Forest 訓練成功")
        
        # 測試 Random Forest
        rf_model = detector.train_random_forest(X_train, y_train)
        print("✓ Random Forest 訓練成功")
        
        # 測試預測
        X_test = np.random.randn(20, 10)
        iso_pred, iso_score = detector.predict_anomaly(iso_model, X_test, 'isolation_forest')
        rf_pred, rf_score = detector.predict_anomaly(rf_model, X_test, 'random_forest')
        
        print("✓ 預測功能正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 異常檢測測試失敗: {e}")
        traceback.print_exc()
        return False

def test_evaluation():
    """測試模型評估"""
    print("\n測試模型評估...")
    
    try:
        from evaluator import ModelEvaluator
        import numpy as np
        
        # 建立測試資料
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)
        
        # 測試評估器
        evaluator = ModelEvaluator()
        result = evaluator.evaluate_model(y_true, y_pred, y_scores, "Test Model")
        
        print("✓ 模型評估成功")
        print(f"✓ F1-Score: {result['f1_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型評估測試失敗: {e}")
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("=== 金融交易異常檢測系統測試 ===\n")
    
    tests = [
        ("模組匯入", test_imports),
        ("資料載入", test_data_loading),
        ("特徵工程", test_feature_engineering),
        ("異常檢測", test_anomaly_detection),
        ("模型評估", test_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"測試: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 測試通過")
            else:
                print(f"✗ {test_name} 測試失敗")
        except Exception as e:
            print(f"✗ {test_name} 測試異常: {e}")
    
    print(f"\n{'='*50}")
    print(f"測試結果: {passed}/{total} 通過")
    print('='*50)
    
    if passed == total:
        print("🎉 所有測試通過！系統準備就緒。")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查相關模組。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
