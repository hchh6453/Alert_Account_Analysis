"""
改進的特徵工程模組
基於警示帳戶分析結果，添加更有效的特徵
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from scipy import stats

class EnhancedFeatureEngineer:
    """改進的特徵工程器"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_enhanced_features(self, account, transactions_df, alerts_df, predict_df):
        """為單一帳戶建立改進的特徵"""
        features = {'acct': account}
        
        # 取得該帳戶的交易資料
        account_txns = self.get_account_transactions(account, transactions_df)
        
        if len(account_txns) == 0:
            # 如果沒有交易資料，建立零特徵
            features.update(self.create_zero_features())
        else:
            # 建立基本特徵 (從原始 FeatureEngineer 繼承)
            features.update(self.create_basic_temporal_features(account_txns))
            features.update(self.create_basic_amount_features(account_txns))
            features.update(self.create_basic_counterparty_features(account_txns))
            features.update(self.create_basic_pattern_features(account_txns))
            features.update(self.create_basic_network_features(account_txns))
            
            # 新增：基於分析結果的特徵
            features.update(self.create_anomaly_features(account_txns))
            features.update(self.create_behavioral_features(account_txns))
            features.update(self.create_risk_features(account_txns))
        
        # 添加標籤
        features['label'] = self.get_account_label(account, alerts_df, predict_df)
        
        return features
    
    def create_basic_temporal_features(self, account_txns):
        """基本時間特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return {
                'total_transactions': 0,
                'transaction_days': 0,
                'avg_transactions_per_day': 0,
                'night_transactions': 0,
                'night_transaction_ratio': 0,
                'weekend_transactions': 0,
                'weekend_transaction_ratio': 0
            }
        
        features['total_transactions'] = len(account_txns)
        
        # 交易天數
        if 'txn_date' in account_txns.columns:
            features['transaction_days'] = account_txns['txn_date'].nunique()
        else:
            features['transaction_days'] = 1
        
        features['avg_transactions_per_day'] = features['total_transactions'] / max(features['transaction_days'], 1)
        
        # 夜間和週末交易
        if 'txn_datetime' in account_txns.columns:
            account_txns['hour'] = account_txns['txn_datetime'].dt.hour
            features['night_transactions'] = len(account_txns[(account_txns['hour'] >= 22) | (account_txns['hour'] <= 6)])
            features['night_transaction_ratio'] = features['night_transactions'] / max(features['total_transactions'], 1)
            
            account_txns['weekday'] = account_txns['txn_datetime'].dt.weekday
            features['weekend_transactions'] = len(account_txns[account_txns['weekday'] >= 5])
            features['weekend_transaction_ratio'] = features['weekend_transactions'] / max(features['total_transactions'], 1)
        else:
            features['night_transactions'] = 0
            features['night_transaction_ratio'] = 0
            features['weekend_transactions'] = 0
            features['weekend_transaction_ratio'] = 0
        
        return features
    
    def create_basic_amount_features(self, account_txns):
        """基本金額特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return {
                'total_amount': 0,
                'avg_amount': 0,
                'max_amount': 0,
                'min_amount': 0,
                'amount_std': 0,
                'amount_cv': 0,
                'outbound_amount': 0,
                'inbound_amount': 0,
                'net_amount': 0,
                'inbound_ratio': 0
            }
        
        amounts = account_txns['txn_amt'].values
        features['total_amount'] = np.sum(amounts)
        features['avg_amount'] = np.mean(amounts)
        features['max_amount'] = np.max(amounts)
        features['min_amount'] = np.min(amounts)
        features['amount_std'] = np.std(amounts)
        features['amount_cv'] = features['amount_std'] / max(features['avg_amount'], 1)
        
        # 轉入轉出金額
        outbound_txns = account_txns[account_txns['direction'] == 'out']
        inbound_txns = account_txns[account_txns['direction'] == 'in']
        
        features['outbound_amount'] = outbound_txns['txn_amt'].sum() if len(outbound_txns) > 0 else 0
        features['inbound_amount'] = inbound_txns['txn_amt'].sum() if len(inbound_txns) > 0 else 0
        features['net_amount'] = features['inbound_amount'] - features['outbound_amount']
        features['inbound_ratio'] = features['inbound_amount'] / max(features['total_amount'], 1)
        
        return features
    
    def create_basic_counterparty_features(self, account_txns):
        """基本交易對手特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return {
                'total_unique_counterparties': 0,
                'unique_outbound_counterparties': 0,
                'unique_inbound_counterparties': 0,
                'repeat_counterparty_ratio': 0,
                'outbound_counterparty_concentration': 0,
                'inbound_counterparty_concentration': 0
            }
        
        # 所有交易對手
        all_counterparties = []
        for _, row in account_txns.iterrows():
            if row['direction'] == 'out':
                all_counterparties.append(row['to_acct'])
            else:
                all_counterparties.append(row['from_acct'])
        
        features['total_unique_counterparties'] = len(set(all_counterparties))
        
        # 轉出交易對手
        outbound_counterparties = account_txns[account_txns['direction'] == 'out']['to_acct'].tolist()
        features['unique_outbound_counterparties'] = len(set(outbound_counterparties))
        
        # 轉入交易對手
        inbound_counterparties = account_txns[account_txns['direction'] == 'in']['from_acct'].tolist()
        features['unique_inbound_counterparties'] = len(set(inbound_counterparties))
        
        # 重複交易對手比例
        counterparty_counts = Counter(all_counterparties)
        repeat_count = sum(1 for count in counterparty_counts.values() if count > 1)
        features['repeat_counterparty_ratio'] = repeat_count / max(len(counterparty_counts), 1)
        
        # 交易對手集中度
        if len(counterparty_counts) > 0:
            max_count = max(counterparty_counts.values())
            features['outbound_counterparty_concentration'] = max_count / len(all_counterparties)
            features['inbound_counterparty_concentration'] = max_count / len(all_counterparties)
        else:
            features['outbound_counterparty_concentration'] = 0
            features['inbound_counterparty_concentration'] = 0
        
        return features
    
    def create_basic_pattern_features(self, account_txns):
        """基本模式特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return {
                'self_transaction_ratio': 0,
                'same_amount_transaction_count': 0,
                'large_transaction_count': 0,
                'large_transaction_ratio': 0,
                'amount_outlier_count': 0,
                'amount_outlier_ratio': 0
            }
        
        # 自轉交易
        self_txns = account_txns[account_txns['from_acct'] == account_txns['to_acct']]
        features['self_transaction_ratio'] = len(self_txns) / max(len(account_txns), 1)
        
        # 相同金額交易
        amounts = account_txns['txn_amt'].values
        amount_counts = Counter(amounts)
        same_amount_count = sum(count for count in amount_counts.values() if count > 1)
        features['same_amount_transaction_count'] = same_amount_count
        
        # 大額交易 (超過平均值的2倍)
        avg_amount = np.mean(amounts)
        large_txns = account_txns[account_txns['txn_amt'] > avg_amount * 2]
        features['large_transaction_count'] = len(large_txns)
        features['large_transaction_ratio'] = len(large_txns) / max(len(account_txns), 1)
        
        # 金額異常值
        if len(amounts) > 2:
            Q1 = np.percentile(amounts, 25)
            Q3 = np.percentile(amounts, 75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 1.5 * IQR
            outliers = account_txns[account_txns['txn_amt'] > outlier_threshold]
            features['amount_outlier_count'] = len(outliers)
            features['amount_outlier_ratio'] = len(outliers) / max(len(account_txns), 1)
        else:
            features['amount_outlier_count'] = 0
            features['amount_outlier_ratio'] = 0
        
        return features
    
    def create_basic_network_features(self, account_txns):
        """基本網絡特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return {
                'transaction_network_density': 0,
                'transaction_network_clustering': 0
            }
        
        # 簡化的網絡特徵
        all_accounts = set(account_txns['from_acct'].unique()) | set(account_txns['to_acct'].unique())
        n_accounts = len(all_accounts)
        
        if n_accounts > 1:
            # 網絡密度 (實際邊數 / 最大可能邊數)
            max_edges = n_accounts * (n_accounts - 1)
            actual_edges = len(account_txns)
            features['transaction_network_density'] = actual_edges / max_edges
            
            # 簡化的聚類係數
            features['transaction_network_clustering'] = actual_edges / max(n_accounts, 1)
        else:
            features['transaction_network_density'] = 0
            features['transaction_network_clustering'] = 0
        
        return features
    
    def create_anomaly_features(self, account_txns):
        """基於分析結果的異常檢測特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return self.create_zero_anomaly_features()
        
        # 1. 金額異常特徵
        amounts = account_txns['txn_amt'].values
        
        # Z-score 異常檢測
        if len(amounts) > 1:
            z_scores = np.abs(stats.zscore(amounts))
            features['amount_zscore_max'] = np.max(z_scores)
            features['amount_zscore_mean'] = np.mean(z_scores)
            features['amount_zscore_std'] = np.std(z_scores)
        else:
            features['amount_zscore_max'] = 0
            features['amount_zscore_mean'] = 0
            features['amount_zscore_std'] = 0
        
        # 2. 交易頻率異常
        if 'txn_datetime' in account_txns.columns and len(account_txns) > 1:
            account_txns_sorted = account_txns.sort_values('txn_datetime')
            intervals = account_txns_sorted['txn_datetime'].diff().dt.total_seconds() / 3600
            intervals = intervals.dropna()
            
            if len(intervals) > 0:
                features['interval_zscore_max'] = np.max(np.abs(stats.zscore(intervals)))
                features['interval_zscore_mean'] = np.mean(np.abs(stats.zscore(intervals)))
            else:
                features['interval_zscore_max'] = 0
                features['interval_zscore_mean'] = 0
        else:
            features['interval_zscore_max'] = 0
            features['interval_zscore_mean'] = 0
        
        # 3. 異常交易模式檢測
        features['suspicious_pattern_score'] = self.calculate_suspicious_pattern_score(account_txns)
        
        return features
    
    def create_behavioral_features(self, account_txns):
        """行為模式特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return self.create_zero_behavioral_features()
        
        # 1. 交易行為一致性
        amounts = account_txns['txn_amt'].values
        if len(amounts) > 1:
            # 金額一致性 (變異係數的倒數)
            cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
            features['amount_consistency'] = 1 / (1 + cv)  # 越高越一致
            
            # 金額分佈的偏度
            features['amount_skewness'] = stats.skew(amounts)
            features['amount_kurtosis'] = stats.kurtosis(amounts)
        else:
            features['amount_consistency'] = 0
            features['amount_skewness'] = 0
            features['amount_kurtosis'] = 0
        
        # 2. 時間行為模式
        if 'txn_datetime' in account_txns.columns and len(account_txns) > 1:
            hours = account_txns['txn_datetime'].dt.hour
            features['hour_entropy'] = self.calculate_entropy(hours)
            features['hour_concentration'] = self.calculate_concentration(hours)
        else:
            features['hour_entropy'] = 0
            features['hour_concentration'] = 0
        
        # 3. 交易對手行為
        counterparties = account_txns['to_acct'].tolist() + account_txns['from_acct'].tolist()
        counterparties = [cp for cp in counterparties if cp != account_txns.iloc[0]['from_acct']]
        
        if len(counterparties) > 0:
            features['counterparty_entropy'] = self.calculate_entropy(counterparties)
            features['counterparty_concentration'] = self.calculate_concentration(counterparties)
        else:
            features['counterparty_entropy'] = 0
            features['counterparty_concentration'] = 0
        
        return features
    
    def create_risk_features(self, account_txns):
        """風險評估特徵"""
        features = {}
        
        if len(account_txns) == 0:
            return self.create_zero_risk_features()
        
        # 1. 基於分析結果的風險分數
        risk_score = 0
        
        # 轉出金額風險
        outbound_amount = account_txns[account_txns['from_acct'] == account_txns.iloc[0]['from_acct']]['txn_amt'].sum()
        if outbound_amount > 10000:  # 高轉出金額
            risk_score += 2
        
        # 交易頻率風險
        if len(account_txns) > 5:  # 高頻交易
            risk_score += 1
        
        # 時間間隔風險
        if 'txn_datetime' in account_txns.columns and len(account_txns) > 1:
            account_txns_sorted = account_txns.sort_values('txn_datetime')
            intervals = account_txns_sorted['txn_datetime'].diff().dt.total_seconds() / 3600
            intervals = intervals.dropna()
            
            if len(intervals) > 0 and np.mean(intervals) < 24:  # 平均間隔小於24小時
                risk_score += 1
        
        # 金額變異風險
        amounts = account_txns['txn_amt'].values
        if len(amounts) > 1:
            cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
            if cv < 0.2:  # 金額變異很小
                risk_score += 1
        
        features['risk_score'] = risk_score
        
        # 2. 組合風險特徵
        features['high_amount_high_frequency'] = 1 if (outbound_amount > 10000 and len(account_txns) > 5) else 0
        features['low_variability_high_frequency'] = 1 if (len(account_txns) > 5 and cv < 0.2) else 0
        
        return features
    
    def calculate_suspicious_pattern_score(self, account_txns):
        """計算可疑模式分數"""
        score = 0
        
        # 1. 自轉交易
        self_txns = account_txns[account_txns['from_acct'] == account_txns['to_acct']]
        if len(self_txns) > 0:
            score += len(self_txns) * 2
        
        # 2. 相同金額交易
        amounts = account_txns['txn_amt'].values
        if len(amounts) > 1:
            amount_counts = Counter(amounts)
            max_count = max(amount_counts.values())
            if max_count > len(amounts) * 0.5:  # 超過50%的交易是相同金額
                score += 3
        
        # 3. 短時間內大量交易
        if 'txn_datetime' in account_txns.columns and len(account_txns) > 1:
            account_txns_sorted = account_txns.sort_values('txn_datetime')
            time_span = (account_txns_sorted['txn_datetime'].max() - 
                        account_txns_sorted['txn_datetime'].min()).total_seconds() / 3600
            
            if time_span < 24 and len(account_txns) > 10:  # 24小時內超過10筆交易
                score += 2
        
        return score
    
    def calculate_entropy(self, values):
        """計算熵值"""
        if len(values) == 0:
            return 0
        
        value_counts = Counter(values)
        total = len(values)
        entropy = 0
        
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def calculate_concentration(self, values):
        """計算集中度"""
        if len(values) == 0:
            return 0
        
        value_counts = Counter(values)
        total = len(values)
        max_count = max(value_counts.values())
        
        return max_count / total
    
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
    
    def get_account_label(self, account, alerts_df, predict_df):
        """取得帳戶標籤"""
        if account in alerts_df['acct'].values:
            return 1
        elif account in predict_df['acct'].values:
            return -1  # 需要預測的帳戶
        else:
            return 0  # 正常帳戶
    
    def create_zero_features(self):
        """建立零特徵"""
        features = {}
        features.update(self.create_basic_temporal_features(pd.DataFrame()))
        features.update(self.create_basic_amount_features(pd.DataFrame()))
        features.update(self.create_basic_counterparty_features(pd.DataFrame()))
        features.update(self.create_basic_pattern_features(pd.DataFrame()))
        features.update(self.create_basic_network_features(pd.DataFrame()))
        features.update(self.create_zero_anomaly_features())
        features.update(self.create_zero_behavioral_features())
        features.update(self.create_zero_risk_features())
        return features
    
    def create_zero_anomaly_features(self):
        """建立零異常特徵"""
        return {
            'amount_zscore_max': 0,
            'amount_zscore_mean': 0,
            'amount_zscore_std': 0,
            'interval_zscore_max': 0,
            'interval_zscore_mean': 0,
            'suspicious_pattern_score': 0
        }
    
    def create_zero_behavioral_features(self):
        """建立零行為特徵"""
        return {
            'amount_consistency': 0,
            'amount_skewness': 0,
            'amount_kurtosis': 0,
            'hour_entropy': 0,
            'hour_concentration': 0,
            'counterparty_entropy': 0,
            'counterparty_concentration': 0
        }
    
    def create_zero_risk_features(self):
        """建立零風險特徵"""
        return {
            'risk_score': 0,
            'high_amount_high_frequency': 0,
            'low_variability_high_frequency': 0
        }
