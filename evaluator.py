"""
模型評估模組
計算 F1-score 和其他評估指標，分析模型效能
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """模型評估器"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, y_true, y_pred, y_scores=None, model_name="Model"):
        """評估單一模型"""
        print(f"\n=== {model_name} 評估結果 ===")
        
        # 基本指標
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"準確率 (Accuracy): {accuracy:.4f}")
        print(f"精確率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n混淆矩陣:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # 詳細分類報告
        print(f"\n詳細分類報告:")
        print(classification_report(y_true, y_pred, target_names=['正常', '警示']))
        
        # AUC 分數 (如果有預測機率)
        if y_scores is not None:
            try:
                auc = roc_auc_score(y_true, y_scores)
                print(f"AUC-ROC: {auc:.4f}")
            except:
                print("無法計算 AUC-ROC")
        
        # 儲存結果
        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'scores': y_scores
        }
        
        if y_scores is not None:
            try:
                result['auc'] = roc_auc_score(y_true, y_scores)
            except:
                result['auc'] = None
        
        self.results[model_name] = result
        
        return result
    
    def compare_models(self, results_dict):
        """比較多個模型"""
        print("\n=== 模型比較 ===")
        
        comparison_df = pd.DataFrame({
            model_name: {
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'AUC': result.get('auc', None)
            }
            for model_name, result in results_dict.items()
        }).T
        
        print(comparison_df.round(4))
        
        # 找出最佳模型
        best_f1_model = comparison_df['F1-Score'].idxmax()
        best_f1_score = comparison_df.loc[best_f1_model, 'F1-Score']
        
        print(f"\n最佳 F1-Score 模型: {best_f1_model}")
        print(f"最佳 F1-Score: {best_f1_score:.4f}")
        
        return comparison_df, best_f1_model
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model"):
        """繪製混淆矩陣"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['正常', '警示'],
                   yticklabels=['正常', '警示'])
        plt.title(f'{model_name} - 混淆矩陣')
        plt.ylabel('實際標籤')
        plt.xlabel('預測標籤')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_scores, model_name="Model"):
        """繪製 ROC 曲線"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='隨機分類器')
        plt.xlabel('假陽性率 (FPR)')
        plt.ylabel('真陽性率 (TPR)')
        plt.title(f'{model_name} - ROC 曲線')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_scores, model_name="Model"):
        """繪製精確率-召回率曲線"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精確率 (Precision)')
        plt.title(f'{model_name} - 精確率-召回率曲線')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'precision_recall_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.show()
    
    def analyze_predictions(self, y_true, y_pred, y_scores=None, feature_names=None, X_test=None):
        """分析預測結果"""
        print("\n=== 預測結果分析 ===")
        
        # 錯誤預測分析
        errors = y_pred != y_true
        error_rate = errors.sum() / len(y_true)
        
        print(f"錯誤預測率: {error_rate:.4f}")
        print(f"錯誤預測數量: {errors.sum()}")
        
        # 假陽性分析
        false_positives = (y_pred == 1) & (y_true == 0)
        print(f"假陽性數量: {false_positives.sum()}")
        
        # 假陰性分析
        false_negatives = (y_pred == 0) & (y_true == 1)
        print(f"假陰性數量: {false_negatives.sum()}")
        
        # 如果有特徵資料，分析錯誤預測的特徵
        if X_test is not None and feature_names is not None:
            self.analyze_error_features(X_test, errors, feature_names)
    
    def analyze_error_features(self, X_test, errors, feature_names):
        """分析錯誤預測的特徵"""
        print("\n=== 錯誤預測特徵分析 ===")
        
        error_data = X_test[errors]
        correct_data = X_test[~errors]
        
        if len(error_data) > 0 and len(correct_data) > 0:
            # 計算特徵差異
            error_means = error_data.mean()
            correct_means = correct_data.mean()
            
            feature_diff = pd.DataFrame({
                'feature': feature_names,
                'error_mean': error_means,
                'correct_mean': correct_means,
                'difference': error_means - correct_means,
                'abs_difference': np.abs(error_means - correct_means)
            }).sort_values('abs_difference', ascending=False)
            
            print("前10個差異最大的特徵:")
            print(feature_diff.head(10))
    
    def calculate_f1_by_threshold(self, y_true, y_scores, thresholds=None):
        """計算不同閾值下的 F1-Score"""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_thresh)
            f1_scores.append(f1)
        
        # 找出最佳閾值
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"最佳閾值: {best_threshold:.2f}")
        print(f"最佳 F1-Score: {best_f1:.4f}")
        
        return thresholds, f1_scores, best_threshold, best_f1
    
    def plot_f1_by_threshold(self, y_true, y_scores, model_name="Model"):
        """繪製不同閾值下的 F1-Score"""
        thresholds, f1_scores, best_threshold, best_f1 = self.calculate_f1_by_threshold(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
        plt.axvline(x=best_threshold, color='r', linestyle='--', 
                   label=f'最佳閾值: {best_threshold:.2f}')
        plt.xlabel('閾值')
        plt.ylabel('F1-Score')
        plt.title(f'{model_name} - F1-Score vs 閾值')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'f1_threshold_{model_name.lower().replace(" ", "_")}.png')
        plt.show()
        
        return best_threshold, best_f1
    
    def generate_evaluation_report(self, results_dict, output_file="evaluation_report.txt"):
        """生成評估報告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== 金融交易異常檢測模型評估報告 ===\n\n")
            
            for model_name, result in results_dict.items():
                f.write(f"模型: {model_name}\n")
                f.write(f"準確率: {result['accuracy']:.4f}\n")
                f.write(f"精確率: {result['precision']:.4f}\n")
                f.write(f"召回率: {result['recall']:.4f}\n")
                f.write(f"F1-Score: {result['f1_score']:.4f}\n")
                if result.get('auc'):
                    f.write(f"AUC-ROC: {result['auc']:.4f}\n")
                f.write("\n")
            
            # 找出最佳模型
            best_model = max(results_dict.keys(), 
                           key=lambda x: results_dict[x]['f1_score'])
            best_f1 = results_dict[best_model]['f1_score']
            
            f.write(f"最佳模型: {best_model}\n")
            f.write(f"最佳 F1-Score: {best_f1:.4f}\n")
        
        print(f"評估報告已儲存至: {output_file}")
