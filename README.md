# 金融交易異常檢測系統

## 🎯 最佳使用方式

### 推薦：增量學習系統
```bash
python incremental_learning.py
```
- **最佳性能**: F1-Score 0.3063
- **完整資料**: 處理所有 443 萬筆交易
- **持續學習**: 每個批次都改進模型
- **模型保存**: 可以中斷後繼續

### 備用：分批處理系統
```bash
python main_batch.py
```
- **完整處理**: 處理所有資料
- **分批處理**: 避免記憶體問題
- **多模型**: 5個無監督學習模型

## 📊 系統演進

| 版本 | 檔案 | F1-Score | 狀態 |
|------|------|----------|------|
| **最新** | `incremental_learning.py` | 0.3063 | ✅ 推薦 |
| 分批處理 | `main_batch.py` | ~0.25 | ✅ 可用 |
| 進階版本 | `main_advanced.py` | ~0.24 | ✅ 可用 |
| 原始版本 | `main_legacy.py` | ~0.20 | 📦 備份 |

## 🔧 核心模組

- `data_loader.py`: 資料載入
- `alert_focused_feature_engineering.py`: 警示導向特徵工程
- `advanced_anomaly_detector.py`: 進階異常檢測器
- `incremental_learning.py`: 增量學習系統

## 📈 性能提升歷程

1. **原始系統**: F1-Score 0.1-0.2
2. **特徵工程**: F1-Score 0.2-0.25
3. **分批處理**: F1-Score 0.24-0.25
4. **增量學習**: F1-Score 0.3063 ⭐

## 🚀 快速開始

```bash
# 運行最佳系統
python incremental_learning.py

# 查看結果
cat submission_incremental.csv
```

## 📁 輸出檔案

- `submission_incremental.csv`: 最終預測結果
- `models/`: 保存的模型檔案
- `alert_account_patterns.csv`: EDA 分析結果