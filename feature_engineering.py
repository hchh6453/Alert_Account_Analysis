"""
特徵工程模組
從交易資料中提取各種特徵，包括時間序列、金額、交易對手等特徵
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

class FeatureEngineer:
    """特徵工程器"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_account_features(self, transactions_df, alerts_df, predict_df):
        """為每個帳戶建立特徵"""
        print("開始特徵工程...")
        
        # 取得所有帳戶
        all_accounts = self.get_all_accounts(transactions_df, alerts_df, predict_df)
        
        # 建立特徵矩陣
        features_list = []
        
        for i, account in enumerate(all_accounts):
            if i % 1000 == 0:
                print(f"處理帳戶 {i+1}/{len(all_accounts)}")
            
            # 為單一帳戶建立特徵
            account_features = self.create_single_account_features(
                account, transactions_df, alerts_df, predict_df
            )
            features_list.append(account_features)
        
        # 合併所有特徵
        features_df = pd.DataFrame(features_list)
        
        print(f"特徵工程完成，共 {len(features_df)} 個帳戶，{len(features_df.columns)} 個特徵")
        
        return features_df
    
    def get_all_accounts(self, transactions_df, alerts_df, predict_df):
        """取得所有帳戶列表"""
        from_accounts = set(transactions_df['from_acct'].unique())
        to_accounts = set(transactions_df['to_acct'].unique())
        alert_accounts = set(alerts_df['acct'].unique())
        predict_accounts = set(predict_df['acct'].unique())
        
        all_accounts = from_accounts.union(to_accounts).union(alert_accounts).union(predict_accounts)
        return list(all_accounts)
    
    def create_single_account_features(self, account, transactions_df, alerts_df, predict_df):
        """為單一帳戶建立特徵"""
        features = {'acct': account}
        
        # 取得該帳戶的交易資料
        account_txns = self.get_account_transactions(account, transactions_df)
        
        if len(account_txns) == 0:
            # 如果沒有交易資料，建立零特徵
            features.update(self.create_zero_features())
        else:
            # 建立各種特徵
            features.update(self.create_temporal_features(account_txns))
            features.update(self.create_amount_features(account_txns))
            features.update(self.create_counterparty_features(account_txns))
            features.update(self.create_pattern_features(account_txns))
            features.update(self.create_network_features(account_txns))
        
        # 添加標籤
        features['label'] = self.get_account_label(account, alerts_df, predict_df)
        
        return features
    
    def get_account_transactions(self, account, transactions_df):
        """取得帳戶的所有交易"""
        # 作為轉出方的交易
        outbound = transactions_df[transactions_df['from_acct'] == account].copy()
        outbound['direction'] = 'out'
        
        # 作為轉入方的交易
        inbound = transactions_df[transactions_df['to_acct'] == account].copy()
        inbound['direction'] = 'in'
        
        # 合併交易
        all_txns = pd.concat([outbound, inbound], ignore_index=True)
        
        # 檢查是否有 txn_datetime 欄位，如果沒有則使用 txn_date
        if 'txn_datetime' in all_txns.columns:
            all_txns = all_txns.sort_values('txn_datetime')
        else:
            all_txns = all_txns.sort_values('txn_date')
        
        return all_txns
    
    def create_temporal_features(self, account_txns):
        """建立時間相關特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return self.create_zero_temporal_features()
        
        # 基本時間統計
        features['total_transactions'] = len(account_txns)
        features['transaction_days'] = account_txns['txn_date'].nunique()
        features['avg_transactions_per_day'] = features['total_transactions'] / max(features['transaction_days'], 1)
        
        # 交易時間分佈
        if 'txn_datetime' in account_txns.columns:
            account_txns['hour'] = account_txns['txn_datetime'].dt.hour
            features['night_transactions'] = len(account_txns[(account_txns['hour'] >= 22) | (account_txns['hour'] <= 6)])
            features['night_transaction_ratio'] = features['night_transactions'] / max(features['total_transactions'], 1)
            
            # 週末交易
            account_txns['weekday'] = account_txns['txn_datetime'].dt.weekday
            features['weekend_transactions'] = len(account_txns[account_txns['weekday'] >= 5])
            features['weekend_transaction_ratio'] = features['weekend_transactions'] / max(features['total_transactions'], 1)
        else:
            # 如果沒有 txn_datetime，使用簡化的時間分析
            features['night_transactions'] = 0
            features['night_transaction_ratio'] = 0
            features['weekend_transactions'] = 0
            features['weekend_transaction_ratio'] = 0
        
        # 交易頻率變化
        daily_counts = account_txns.groupby('txn_date').size()
        features['transaction_frequency_std'] = daily_counts.std() if len(daily_counts) > 1 else 0
        features['transaction_frequency_cv'] = features['transaction_frequency_std'] / max(daily_counts.mean(), 1)
        
        # 交易間隔
        if len(account_txns) > 1 and 'txn_datetime' in account_txns.columns:
            account_txns = account_txns.sort_values('txn_datetime')
            intervals = account_txns['txn_datetime'].diff().dt.total_seconds() / 3600  # 小時
            features['avg_transaction_interval_hours'] = intervals.mean()
            features['min_transaction_interval_hours'] = intervals.min()
            features['transaction_interval_std'] = intervals.std()
        else:
            features['avg_transaction_interval_hours'] = 0
            features['min_transaction_interval_hours'] = 0
            features['transaction_interval_std'] = 0
        
        return features
    
    def create_amount_features(self, account_txns):
        """建立金額相關特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return self.create_zero_amount_features()
        
        # 基本金額統計
        features['total_amount'] = account_txns['txn_amt'].sum()
        features['avg_amount'] = account_txns['txn_amt'].mean()
        features['max_amount'] = account_txns['txn_amt'].max()
        features['min_amount'] = account_txns['txn_amt'].min()
        features['amount_std'] = account_txns['txn_amt'].std()
        features['amount_cv'] = features['amount_std'] / max(features['avg_amount'], 1)
        
        # 金額分佈
        features['large_transaction_count'] = len(account_txns[account_txns['txn_amt'] > account_txns['txn_amt'].quantile(0.95)])
        features['large_transaction_ratio'] = features['large_transaction_count'] / max(len(account_txns), 1)
        
        # 轉入轉出分析
        inbound_txns = account_txns[account_txns['direction'] == 'in']
        outbound_txns = account_txns[account_txns['direction'] == 'out']
        
        features['inbound_amount'] = inbound_txns['txn_amt'].sum() if len(inbound_txns) > 0 else 0
        features['outbound_amount'] = outbound_txns['txn_amt'].sum() if len(outbound_txns) > 0 else 0
        features['net_amount'] = features['inbound_amount'] - features['outbound_amount']
        features['inbound_ratio'] = features['inbound_amount'] / max(features['total_amount'], 1)
        
        # 金額異常檢測
        features['amount_outlier_count'] = self.detect_amount_outliers(account_txns['txn_amt'])
        features['amount_outlier_ratio'] = features['amount_outlier_count'] / max(len(account_txns), 1)
        
        return features
    
    def create_counterparty_features(self, account_txns):
        """建立交易對手相關特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return self.create_zero_counterparty_features()
        
        # 交易對手統計
        inbound_counterparties = account_txns[account_txns['direction'] == 'in']['from_acct'].nunique()
        outbound_counterparties = account_txns[account_txns['direction'] == 'out']['to_acct'].nunique()
        
        features['unique_inbound_counterparties'] = inbound_counterparties
        features['unique_outbound_counterparties'] = outbound_counterparties
        features['total_unique_counterparties'] = inbound_counterparties + outbound_counterparties
        
        # 交易對手集中度
        inbound_counts = account_txns[account_txns['direction'] == 'in']['from_acct'].value_counts()
        outbound_counts = account_txns[account_txns['direction'] == 'out']['to_acct'].value_counts()
        
        if len(inbound_counts) > 0:
            features['inbound_counterparty_concentration'] = inbound_counts.max() / inbound_counts.sum()
        else:
            features['inbound_counterparty_concentration'] = 0
            
        if len(outbound_counts) > 0:
            features['outbound_counterparty_concentration'] = outbound_counts.max() / outbound_counts.sum()
        else:
            features['outbound_counterparty_concentration'] = 0
        
        # 重複交易對手
        features['repeat_counterparty_ratio'] = self.calculate_repeat_counterparty_ratio(account_txns)
        
        # 交易對手類型分析
        features['self_transaction_ratio'] = len(account_txns[account_txns['is_self_txn'] == 'Y']) / max(len(account_txns), 1)
        
        return features
    
    def create_pattern_features(self, account_txns):
        """建立交易模式特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return self.create_zero_pattern_features()
        
        # 交易模式檢測
        features['burst_transaction_count'] = self.detect_burst_transactions(account_txns)
        features['burst_transaction_ratio'] = features['burst_transaction_count'] / max(len(account_txns), 1)
        
        # 相同金額交易
        features['same_amount_transaction_count'] = self.detect_same_amount_transactions(account_txns)
        features['same_amount_transaction_ratio'] = features['same_amount_transaction_count'] / max(len(account_txns), 1)
        
        # 交易時間模式
        features['regular_time_pattern'] = self.detect_regular_time_pattern(account_txns)
        
        return features
    
    def create_network_features(self, account_txns):
        """建立網路相關特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return self.create_zero_network_features()
        
        # 交易網路特徵
        features['transaction_network_density'] = self.calculate_network_density(account_txns)
        features['transaction_network_clustering'] = self.calculate_network_clustering(account_txns)
        
        return features
    
    def detect_amount_outliers(self, amounts):
        """檢測金額異常值"""
        if len(amounts) < 3:
            return 0
        
        Q1 = amounts.quantile(0.25)
        Q3 = amounts.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = amounts[(amounts < lower_bound) | (amounts > upper_bound)]
        return len(outliers)
    
    def calculate_repeat_counterparty_ratio(self, account_txns):
        """計算重複交易對手比例"""
        inbound_counterparties = account_txns[account_txns['direction'] == 'in']['from_acct']
        outbound_counterparties = account_txns[account_txns['direction'] == 'out']['to_acct']
        
        all_counterparties = pd.concat([inbound_counterparties, outbound_counterparties])
        unique_counterparties = all_counterparties.nunique()
        total_transactions = len(all_counterparties)
        
        if total_transactions == 0:
            return 0
        
        return (total_transactions - unique_counterparties) / total_transactions
    
    def detect_burst_transactions(self, account_txns):
        """檢測爆發性交易"""
        if len(account_txns) < 2:
            return 0
        
        if 'txn_datetime' not in account_txns.columns:
            return 0
        
        # 按小時分組，計算每小時交易數
        account_txns['hour_group'] = account_txns['txn_datetime'].dt.floor('H')
        hourly_counts = account_txns.groupby('hour_group').size()
        
        # 檢測異常高的交易頻率
        mean_count = hourly_counts.mean()
        std_count = hourly_counts.std()
        
        if std_count == 0:
            return 0
        
        burst_threshold = mean_count + 2 * std_count
        burst_hours = hourly_counts[hourly_counts > burst_threshold]
        
        return burst_hours.sum()
    
    def detect_same_amount_transactions(self, account_txns):
        """檢測相同金額交易"""
        amount_counts = account_txns['txn_amt'].value_counts()
        same_amount_txns = amount_counts[amount_counts > 1]
        
        return same_amount_txns.sum()
    
    def detect_regular_time_pattern(self, account_txns):
        """檢測規律時間模式"""
        if len(account_txns) < 3:
            return 0
        
        if 'txn_datetime' not in account_txns.columns:
            return 0
        
        # 計算交易時間的規律性
        hours = account_txns['txn_datetime'].dt.hour
        hour_counts = hours.value_counts()
        
        # 如果某些小時的交易數明顯多於其他小時，則認為有規律模式
        max_count = hour_counts.max()
        total_count = len(account_txns)
        
        return max_count / total_count
    
    def calculate_network_density(self, account_txns):
        """計算交易網路密度"""
        inbound_counterparties = set(account_txns[account_txns['direction'] == 'in']['from_acct'])
        outbound_counterparties = set(account_txns[account_txns['direction'] == 'out']['to_acct'])
        
        all_counterparties = inbound_counterparties.union(outbound_counterparties)
        n = len(all_counterparties)
        
        if n <= 1:
            return 0
        
        # 簡化的密度計算
        max_possible_connections = n * (n - 1) / 2
        actual_connections = len(account_txns)
        
        return actual_connections / max_possible_connections
    
    def calculate_network_clustering(self, account_txns):
        """計算交易網路聚類係數"""
        # 簡化的聚類係數計算
        inbound_counterparties = account_txns[account_txns['direction'] == 'in']['from_acct'].unique()
        outbound_counterparties = account_txns[account_txns['direction'] == 'out']['to_acct'].unique()
        
        # 計算重疊的交易對手
        overlap = len(set(inbound_counterparties).intersection(set(outbound_counterparties)))
        total_unique = len(set(inbound_counterparties).union(set(outbound_counterparties)))
        
        if total_unique == 0:
            return 0
        
        return overlap / total_unique
    
    def get_account_label(self, account, alerts_df, predict_df):
        """取得帳戶標籤"""
        # 檢查是否為警示帳戶
        if account in alerts_df['acct'].values:
            return 1
        
        # 檢查預測資料中的標籤
        predict_row = predict_df[predict_df['acct'] == account]
        if len(predict_row) > 0:
            return predict_row.iloc[0]['label']
        
        # 沒有標籤
        return np.nan
    
    # 零特徵生成方法
    def create_zero_features(self):
        """建立零特徵"""
        features = {}
        features.update(self.create_zero_temporal_features())
        features.update(self.create_zero_amount_features())
        features.update(self.create_zero_counterparty_features())
        features.update(self.create_zero_pattern_features())
        features.update(self.create_zero_network_features())
        return features
    
    def create_zero_temporal_features(self):
        return {
            'total_transactions': 0,
            'transaction_days': 0,
            'avg_transactions_per_day': 0,
            'night_transactions': 0,
            'night_transaction_ratio': 0,
            'weekend_transactions': 0,
            'weekend_transaction_ratio': 0,
            'transaction_frequency_std': 0,
            'transaction_frequency_cv': 0,
            'avg_transaction_interval_hours': 0,
            'min_transaction_interval_hours': 0,
            'transaction_interval_std': 0
        }
    
    def create_zero_amount_features(self):
        return {
            'total_amount': 0,
            'avg_amount': 0,
            'max_amount': 0,
            'min_amount': 0,
            'amount_std': 0,
            'amount_cv': 0,
            'large_transaction_count': 0,
            'large_transaction_ratio': 0,
            'inbound_amount': 0,
            'outbound_amount': 0,
            'net_amount': 0,
            'inbound_ratio': 0,
            'amount_outlier_count': 0,
            'amount_outlier_ratio': 0
        }
    
    def create_zero_counterparty_features(self):
        return {
            'unique_inbound_counterparties': 0,
            'unique_outbound_counterparties': 0,
            'total_unique_counterparties': 0,
            'inbound_counterparty_concentration': 0,
            'outbound_counterparty_concentration': 0,
            'repeat_counterparty_ratio': 0,
            'self_transaction_ratio': 0
        }
    
    def create_zero_pattern_features(self):
        return {
            'burst_transaction_count': 0,
            'burst_transaction_ratio': 0,
            'same_amount_transaction_count': 0,
            'same_amount_transaction_ratio': 0,
            'regular_time_pattern': 0
        }
    
    def create_zero_network_features(self):
        return {
            'transaction_network_density': 0,
            'transaction_network_clustering': 0
        }
