"""
警示帳戶深度 EDA 分析
對已標記的警示帳戶進行全面的探索性資料分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AlertAccountEDA:
    """警示帳戶探索性資料分析"""
    
    def __init__(self):
        self.alert_accounts = None
        self.normal_accounts = None
        self.transactions_df = None
        
    def load_and_prepare_data(self, transactions_df, alerts_df, sample_size=1000):
        """載入並準備資料"""
        print("=== 載入警示帳戶資料 ===")
        
        # 取得警示帳戶
        alert_accounts = set(alerts_df['acct'])
        print(f"警示帳戶總數: {len(alert_accounts)}")
        
        # 取得正常帳戶 (隨機取樣)
        from_accounts = set(transactions_df['from_acct'].unique())
        to_accounts = set(transactions_df['to_acct'].unique())
        all_accounts = list(from_accounts.union(to_accounts))
        normal_accounts = [acc for acc in all_accounts if acc not in alert_accounts]
        
        # 隨機取樣正常帳戶
        import random
        random.seed(42)
        normal_sample = random.sample(normal_accounts, min(sample_size, len(normal_accounts)))
        
        print(f"正常帳戶樣本: {len(normal_sample)}")
        
        self.alert_accounts = alert_accounts
        self.normal_accounts = set(normal_sample)
        self.transactions_df = transactions_df
        
        return alert_accounts, normal_sample
    
    def analyze_transaction_patterns(self):
        """分析交易模式"""
        print("\n=== 交易模式分析 ===")
        
        alert_patterns = []
        normal_patterns = []
        
        # 分析警示帳戶
        for account in list(self.alert_accounts)[:50]:  # 限制分析數量
            account_txns = self.get_account_transactions(account)
            if len(account_txns) > 0:
                pattern = self.extract_transaction_pattern(account_txns)
                pattern['account_type'] = 'alert'
                alert_patterns.append(pattern)
        
        # 分析正常帳戶
        for account in list(self.normal_accounts)[:200]:  # 限制分析數量
            account_txns = self.get_account_transactions(account)
            if len(account_txns) > 0:
                pattern = self.extract_transaction_pattern(account_txns)
                pattern['account_type'] = 'normal'
                normal_patterns.append(pattern)
        
        # 合併資料
        all_patterns = alert_patterns + normal_patterns
        patterns_df = pd.DataFrame(all_patterns)
        
        return patterns_df
    
    def extract_transaction_pattern(self, account_txns):
        """提取交易模式"""
        pattern = {}
        
        # 基本統計
        pattern['total_transactions'] = len(account_txns)
        pattern['total_amount'] = account_txns['txn_amt'].sum()
        pattern['avg_amount'] = account_txns['txn_amt'].mean()
        pattern['max_amount'] = account_txns['txn_amt'].max()
        pattern['min_amount'] = account_txns['txn_amt'].min()
        pattern['amount_std'] = account_txns['txn_amt'].std()
        
        # 轉入轉出分析
        outbound = account_txns[account_txns['direction'] == 'out']
        inbound = account_txns[account_txns['direction'] == 'in']
        
        pattern['outbound_count'] = len(outbound)
        pattern['inbound_count'] = len(inbound)
        pattern['outbound_amount'] = outbound['txn_amt'].sum() if len(outbound) > 0 else 0
        pattern['inbound_amount'] = inbound['txn_amt'].sum() if len(inbound) > 0 else 0
        pattern['net_amount'] = pattern['inbound_amount'] - pattern['outbound_amount']
        
        # 交易對手分析
        outbound_counterparties = outbound['to_acct'].tolist()
        inbound_counterparties = inbound['from_acct'].tolist()
        
        pattern['unique_outbound_counterparties'] = len(set(outbound_counterparties))
        pattern['unique_inbound_counterparties'] = len(set(inbound_counterparties))
        pattern['counterparty_diversity'] = len(set(outbound_counterparties + inbound_counterparties))
        
        # 時間模式分析
        if 'txn_datetime' in account_txns.columns and len(account_txns) > 1:
            account_txns_sorted = account_txns.sort_values('txn_datetime')
            intervals = account_txns_sorted['txn_datetime'].diff().dt.total_seconds() / 3600
            intervals = intervals.dropna()
            
            pattern['avg_interval_hours'] = intervals.mean() if len(intervals) > 0 else 0
            pattern['min_interval_hours'] = intervals.min() if len(intervals) > 0 else 0
            pattern['max_interval_hours'] = intervals.max() if len(intervals) > 0 else 0
            pattern['interval_std'] = intervals.std() if len(intervals) > 0 else 0
            
            # 時間分佈
            hours = account_txns['txn_datetime'].dt.hour
            pattern['night_transactions'] = len(account_txns[(hours >= 22) | (hours <= 6)])
            pattern['weekend_transactions'] = len(account_txns[account_txns['txn_datetime'].dt.weekday >= 5])
        else:
            pattern['avg_interval_hours'] = 0
            pattern['min_interval_hours'] = 0
            pattern['max_interval_hours'] = 0
            pattern['interval_std'] = 0
            pattern['night_transactions'] = 0
            pattern['weekend_transactions'] = 0
        
        # 金額模式分析
        amounts = account_txns['txn_amt'].values
        pattern['amount_cv'] = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
        
        # 相同金額交易
        amount_counts = Counter(amounts)
        same_amount_count = sum(count for count in amount_counts.values() if count > 1)
        pattern['same_amount_transactions'] = same_amount_count
        pattern['same_amount_ratio'] = same_amount_count / len(amounts)
        
        # 自轉交易
        self_txns = account_txns[account_txns['from_acct'] == account_txns['to_acct']]
        pattern['self_transactions'] = len(self_txns)
        pattern['self_transaction_ratio'] = len(self_txns) / len(account_txns)
        
        return pattern
    
    def get_account_transactions(self, account):
        """取得帳戶的所有交易"""
        # 作為轉出方的交易
        outbound = self.transactions_df[self.transactions_df['from_acct'] == account].copy()
        outbound['direction'] = 'out'
        
        # 作為轉入方的交易
        inbound = self.transactions_df[self.transactions_df['to_acct'] == account].copy()
        inbound['direction'] = 'in'
        
        # 合併交易
        all_txns = pd.concat([outbound, inbound], ignore_index=True)
        
        # 檢查是否有 txn_datetime 欄位，如果沒有則使用 txn_date
        if 'txn_datetime' in all_txns.columns:
            all_txns = all_txns.sort_values('txn_datetime')
        else:
            all_txns = all_txns.sort_values('txn_date')
        
        return all_txns
    
    def analyze_amount_distributions(self, patterns_df):
        """分析金額分佈"""
        print("\n=== 金額分佈分析 ===")
        
        alert_data = patterns_df[patterns_df['account_type'] == 'alert']
        normal_data = patterns_df[patterns_df['account_type'] == 'normal']
        
        print(f"警示帳戶樣本: {len(alert_data)}")
        print(f"正常帳戶樣本: {len(normal_data)}")
        
        # 金額統計
        print("\n金額統計比較:")
        print("警示帳戶:")
        print(f"  平均金額: {alert_data['avg_amount'].mean():.2f}")
        print(f"  最大金額: {alert_data['max_amount'].mean():.2f}")
        print(f"  金額變異係數: {alert_data['amount_cv'].mean():.4f}")
        print(f"  總交易金額: {alert_data['total_amount'].mean():.2f}")
        
        print("\n正常帳戶:")
        print(f"  平均金額: {normal_data['avg_amount'].mean():.2f}")
        print(f"  最大金額: {normal_data['max_amount'].mean():.2f}")
        print(f"  金額變異係數: {normal_data['amount_cv'].mean():.4f}")
        print(f"  總交易金額: {normal_data['total_amount'].mean():.2f}")
        
        return alert_data, normal_data
    
    def analyze_transaction_frequency(self, patterns_df):
        """分析交易頻率"""
        print("\n=== 交易頻率分析 ===")
        
        alert_data = patterns_df[patterns_df['account_type'] == 'alert']
        normal_data = patterns_df[patterns_df['account_type'] == 'normal']
        
        print("交易頻率比較:")
        print("警示帳戶:")
        print(f"  平均交易數: {alert_data['total_transactions'].mean():.2f}")
        print(f"  平均間隔: {alert_data['avg_interval_hours'].mean():.2f} 小時")
        print(f"  最短間隔: {alert_data['min_interval_hours'].mean():.2f} 小時")
        
        print("\n正常帳戶:")
        print(f"  平均交易數: {normal_data['total_transactions'].mean():.2f}")
        print(f"  平均間隔: {normal_data['avg_interval_hours'].mean():.2f} 小時")
        print(f"  最短間隔: {normal_data['min_interval_hours'].mean():.2f} 小時")
    
    def analyze_behavioral_patterns(self, patterns_df):
        """分析行為模式"""
        print("\n=== 行為模式分析 ===")
        
        alert_data = patterns_df[patterns_df['account_type'] == 'alert']
        normal_data = patterns_df[patterns_df['account_type'] == 'normal']
        
        print("行為模式比較:")
        print("警示帳戶:")
        print(f"  自轉交易比例: {alert_data['self_transaction_ratio'].mean():.4f}")
        print(f"  相同金額交易比例: {alert_data['same_amount_ratio'].mean():.4f}")
        print(f"  淨流出比例: {(alert_data['net_amount'] < 0).mean():.4f}")
        print(f"  交易對手多樣性: {alert_data['counterparty_diversity'].mean():.2f}")
        
        print("\n正常帳戶:")
        print(f"  自轉交易比例: {normal_data['self_transaction_ratio'].mean():.4f}")
        print(f"  相同金額交易比例: {normal_data['same_amount_ratio'].mean():.4f}")
        print(f"  淨流出比例: {(normal_data['net_amount'] < 0).mean():.4f}")
        print(f"  交易對手多樣性: {normal_data['counterparty_diversity'].mean():.2f}")
    
    def identify_key_differences(self, patterns_df):
        """識別關鍵差異"""
        print("\n=== 關鍵差異識別 ===")
        
        alert_data = patterns_df[patterns_df['account_type'] == 'alert']
        normal_data = patterns_df[patterns_df['account_type'] == 'normal']
        
        # 計算差異倍數
        differences = {}
        numeric_cols = patterns_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'account_type']
        
        for col in numeric_cols:
            alert_mean = alert_data[col].mean()
            normal_mean = normal_data[col].mean()
            
            if normal_mean != 0:
                ratio = alert_mean / normal_mean
                differences[col] = {
                    'alert_mean': alert_mean,
                    'normal_mean': normal_mean,
                    'ratio': ratio,
                    'abs_ratio': abs(ratio - 1)
                }
        
        # 排序並顯示最重要的差異
        sorted_diffs = sorted(differences.items(), key=lambda x: x[1]['abs_ratio'], reverse=True)
        
        print("最重要的差異 (前10個):")
        for i, (col, diff) in enumerate(sorted_diffs[:10]):
            print(f"{i+1:2d}. {col:30s} | 警示: {diff['alert_mean']:8.2f} | 正常: {diff['normal_mean']:8.2f} | 倍數: {diff['ratio']:6.2f}")
        
        return differences
    
    def suggest_new_features(self, differences):
        """建議新特徵"""
        print("\n=== 建議新特徵 ===")
        
        # 基於差異分析建議新特徵
        suggestions = []
        
        for col, diff in differences.items():
            if diff['abs_ratio'] > 1.5:  # 差異超過50%
                suggestions.append({
                    'feature': col,
                    'difference': diff['abs_ratio'],
                    'suggestion': f"基於 {col} 的差異，可以設計相關的風險特徵"
                })
        
        print("基於分析結果，建議以下新特徵:")
        for i, suggestion in enumerate(suggestions[:10]):
            print(f"{i+1:2d}. {suggestion['feature']:30s} (差異: {suggestion['difference']:.2f})")
        
        return suggestions
    
    def run_complete_eda(self, transactions_df, alerts_df):
        """執行完整的 EDA 分析"""
        print("=== 警示帳戶完整 EDA 分析 ===")
        
        # 1. 載入資料
        alert_accounts, normal_accounts = self.load_and_prepare_data(transactions_df, alerts_df)
        
        # 2. 分析交易模式
        patterns_df = self.analyze_transaction_patterns()
        
        # 3. 分析金額分佈
        alert_data, normal_data = self.analyze_amount_distributions(patterns_df)
        
        # 4. 分析交易頻率
        self.analyze_transaction_frequency(patterns_df)
        
        # 5. 分析行為模式
        self.analyze_behavioral_patterns(patterns_df)
        
        # 6. 識別關鍵差異
        differences = self.identify_key_differences(patterns_df)
        
        # 7. 建議新特徵
        suggestions = self.suggest_new_features(differences)
        
        # 8. 保存結果
        patterns_df.to_csv('alert_account_patterns.csv', index=False)
        
        print(f"\n=== EDA 分析完成 ===")
        print(f"分析結果已保存至 alert_account_patterns.csv")
        
        return patterns_df, differences, suggestions

def run_alert_eda():
    """執行警示帳戶 EDA"""
    from data_loader import DataLoader
    
    # 載入資料
    print("載入資料...")
    data_loader = DataLoader()
    transactions_df, alerts_df, predict_df = data_loader.load_data()
    
    # 限制資料量
    transactions_df = transactions_df.head(1000000)  # 100萬筆交易
    
    # 執行 EDA
    eda = AlertAccountEDA()
    patterns_df, differences, suggestions = eda.run_complete_eda(transactions_df, alerts_df)
    
    return patterns_df, differences, suggestions

if __name__ == "__main__":
    run_alert_eda()
