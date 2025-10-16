"""
專門針對警示帳戶的特徵工程
基於真實警示帳戶模式設計特徵
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from scipy import stats

class AlertFocusedFeatureEngineer:
    """專門針對警示帳戶的特徵工程器"""
    
    def __init__(self):
        self.feature_names = []
        # 基於分析結果的警示帳戶特徵閾值
        self.thresholds = {
            'high_outbound_amount': 10000,  # 高轉出金額閾值
            'short_interval_hours': 24,     # 短間隔閾值
            'low_amount_cv': 0.3,           # 低變異係數閾值
            'high_frequency': 5,             # 高頻交易閾值
            'suspicious_pattern_score': 2   # 可疑模式分數閾值
        }
    
    def create_alert_focused_features(self, account, transactions_df, alerts_df, predict_df):
        """為單一帳戶建立警示導向的特徵"""
        features = {'acct': account}
        
        # 取得該帳戶的交易資料
        account_txns = self.get_account_transactions(account, transactions_df)
        
        if len(account_txns) == 0:
            features.update(self.create_zero_alert_features())
        else:
            # 建立警示導向特徵
            features.update(self.create_alert_risk_features(account_txns))
            features.update(self.create_alert_behavior_features(account_txns))
            features.update(self.create_alert_pattern_features(account_txns))
            features.update(self.create_alert_anomaly_features(account_txns))
            features.update(self.create_alert_composite_features(account_txns))
        
        # 添加標籤
        features['label'] = self.get_account_label(account, alerts_df, predict_df)
        
        return features
    
    def create_alert_risk_features(self, account_txns):
        """警示風險特徵"""
        features = {}
        
        # 1. 轉出金額風險
        outbound_amount = account_txns[account_txns['direction'] == 'out']['txn_amt'].sum()
        features['outbound_amount_risk'] = 1 if outbound_amount > self.thresholds['high_outbound_amount'] else 0
        features['outbound_amount_ratio'] = outbound_amount / max(account_txns['txn_amt'].sum(), 1)
        
        # 2. 交易頻率風險
        features['high_frequency_risk'] = 1 if len(account_txns) > self.thresholds['high_frequency'] else 0
        features['transaction_frequency'] = len(account_txns)
        
        # 3. 時間間隔風險
        if 'txn_datetime' in account_txns.columns and len(account_txns) > 1:
            account_txns_sorted = account_txns.sort_values('txn_datetime')
            intervals = account_txns_sorted['txn_datetime'].diff().dt.total_seconds() / 3600
            intervals = intervals.dropna()
            
            if len(intervals) > 0:
                avg_interval = np.mean(intervals)
                features['short_interval_risk'] = 1 if avg_interval < self.thresholds['short_interval_hours'] else 0
                features['avg_interval_hours'] = avg_interval
            else:
                features['short_interval_risk'] = 0
                features['avg_interval_hours'] = 0
        else:
            features['short_interval_risk'] = 0
            features['avg_interval_hours'] = 0
        
        # 4. 金額變異風險
        amounts = account_txns['txn_amt'].values
        if len(amounts) > 1:
            cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
            features['low_variability_risk'] = 1 if cv < self.thresholds['low_amount_cv'] else 0
            features['amount_cv'] = cv
        else:
            features['low_variability_risk'] = 0
            features['amount_cv'] = 0
        
        return features
    
    def create_alert_behavior_features(self, account_txns):
        """警示行為特徵"""
        features = {}
        
        # 1. 資金流向特徵
        inbound_amount = account_txns[account_txns['direction'] == 'in']['txn_amt'].sum()
        outbound_amount = account_txns[account_txns['direction'] == 'out']['txn_amt'].sum()
        
        features['net_outflow'] = 1 if outbound_amount > inbound_amount else 0
        features['outbound_dominance'] = outbound_amount / max(inbound_amount + outbound_amount, 1)
        
        # 2. 交易對手行為
        outbound_counterparties = account_txns[account_txns['direction'] == 'out']['to_acct'].tolist()
        inbound_counterparties = account_txns[account_txns['direction'] == 'in']['from_acct'].tolist()
        
        features['outbound_counterparty_diversity'] = len(set(outbound_counterparties))
        features['inbound_counterparty_diversity'] = len(set(inbound_counterparties))
        features['counterparty_imbalance'] = len(set(outbound_counterparties)) - len(set(inbound_counterparties))
        
        # 3. 時間行為 (基於 EDA 發現的重要模式)
        if 'txn_datetime' in account_txns.columns:
            hours = account_txns['txn_datetime'].dt.hour
            weekdays = account_txns['txn_datetime'].dt.weekday
            
            # 夜間交易 (EDA 發現: 3.59倍差異)
            night_txns = len(account_txns[(hours >= 22) | (hours <= 6)])
            features['night_transaction_ratio'] = night_txns / len(account_txns)
            features['night_transaction_count'] = night_txns
            
            # 週末交易 (EDA 發現: 4.77倍差異 - 最重要!)
            weekend_txns = len(account_txns[weekdays >= 5])
            features['weekend_transaction_ratio'] = weekend_txns / len(account_txns)
            features['weekend_transaction_count'] = weekend_txns
            
            # 交易間隔變異性 (EDA 發現: 4.60倍差異 - 第二重要!)
            if len(account_txns) > 1:
                account_txns_sorted = account_txns.sort_values('txn_datetime')
                intervals = account_txns_sorted['txn_datetime'].diff().dt.total_seconds() / 3600
                intervals = intervals.dropna()
                
                if len(intervals) > 0:
                    features['interval_variability'] = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
                    features['interval_std'] = intervals.std()
                else:
                    features['interval_variability'] = 0
                    features['interval_std'] = 0
            else:
                features['interval_variability'] = 0
                features['interval_std'] = 0
        else:
            features['night_transaction_ratio'] = 0
            features['night_transaction_count'] = 0
            features['weekend_transaction_ratio'] = 0
            features['weekend_transaction_count'] = 0
            features['interval_variability'] = 0
            features['interval_std'] = 0
        
        return features
    
    def create_alert_pattern_features(self, account_txns):
        """警示模式特徵"""
        features = {}
        
        # 1. 自轉交易
        self_txns = account_txns[account_txns['from_acct'] == account_txns['to_acct']]
        features['self_transaction_count'] = len(self_txns)
        features['self_transaction_ratio'] = len(self_txns) / len(account_txns)
        
        # 2. 相同金額交易 (EDA 發現: 4.55倍差異 - 第三重要!)
        amounts = account_txns['txn_amt'].values
        amount_counts = Counter(amounts)
        same_amount_count = sum(count for count in amount_counts.values() if count > 1)
        features['same_amount_transaction_count'] = same_amount_count
        features['same_amount_transaction_ratio'] = same_amount_count / len(account_txns)
        
        # 相同金額交易頻率 (EDA 新發現)
        features['same_amount_frequency'] = same_amount_count / max(len(amounts), 1)
        
        # 3. 金額分佈特徵
        if len(amounts) > 1:
            features['amount_skewness'] = stats.skew(amounts)
            features['amount_kurtosis'] = stats.kurtosis(amounts)
            
            # 金額集中度
            max_amount_count = max(amount_counts.values())
            features['amount_concentration'] = max_amount_count / len(amounts)
        else:
            features['amount_skewness'] = 0
            features['amount_kurtosis'] = 0
            features['amount_concentration'] = 0
        
        # 4. 交易時間模式
        if 'txn_datetime' in account_txns.columns and len(account_txns) > 1:
            account_txns_sorted = account_txns.sort_values('txn_datetime')
            time_span = (account_txns_sorted['txn_datetime'].max() - 
                        account_txns_sorted['txn_datetime'].min()).total_seconds() / 3600
            
            features['transaction_time_span_hours'] = time_span
            features['transaction_intensity'] = len(account_txns) / max(time_span, 1)  # 每小時交易數
        else:
            features['transaction_time_span_hours'] = 0
            features['transaction_intensity'] = 0
        
        return features
    
    def create_alert_anomaly_features(self, account_txns):
        """警示異常特徵"""
        features = {}
        
        # 1. 金額異常檢測
        amounts = account_txns['txn_amt'].values
        if len(amounts) > 2:
            z_scores = np.abs(stats.zscore(amounts))
            features['amount_zscore_max'] = np.max(z_scores)
            features['amount_zscore_mean'] = np.mean(z_scores)
            
            # 異常值比例
            outlier_threshold = 2.0
            outliers = np.sum(z_scores > outlier_threshold)
            features['amount_outlier_ratio'] = outliers / len(amounts)
        else:
            features['amount_zscore_max'] = 0
            features['amount_zscore_mean'] = 0
            features['amount_outlier_ratio'] = 0
        
        # 2. 時間異常檢測
        if 'txn_datetime' in account_txns.columns and len(account_txns) > 2:
            account_txns_sorted = account_txns.sort_values('txn_datetime')
            intervals = account_txns_sorted['txn_datetime'].diff().dt.total_seconds() / 3600
            intervals = intervals.dropna()
            
            if len(intervals) > 0:
                interval_z_scores = np.abs(stats.zscore(intervals))
                features['interval_zscore_max'] = np.max(interval_z_scores)
                features['interval_zscore_mean'] = np.mean(interval_z_scores)
            else:
                features['interval_zscore_max'] = 0
                features['interval_zscore_mean'] = 0
        else:
            features['interval_zscore_max'] = 0
            features['interval_zscore_mean'] = 0
        
        # 3. 可疑模式分數
        features['suspicious_pattern_score'] = self.calculate_suspicious_pattern_score(account_txns)
        
        return features
    
    def create_alert_composite_features(self, account_txns):
        """警示組合特徵"""
        features = {}
        
        # 1. 高風險組合
        outbound_amount = account_txns[account_txns['direction'] == 'out']['txn_amt'].sum()
        high_amount = 1 if outbound_amount > self.thresholds['high_outbound_amount'] else 0
        high_frequency = 1 if len(account_txns) > self.thresholds['high_frequency'] else 0
        
        features['high_amount_high_frequency'] = high_amount * high_frequency
        
        # 2. 可疑行為組合
        amounts = account_txns['txn_amt'].values
        cv = np.std(amounts) / np.mean(amounts) if len(amounts) > 1 and np.mean(amounts) > 0 else 0
        low_variability = 1 if cv < self.thresholds['low_amount_cv'] else 0
        
        features['low_variability_high_frequency'] = low_variability * high_frequency
        
        # 3. 綜合風險分數
        risk_score = 0
        risk_score += high_amount
        risk_score += high_frequency
        risk_score += low_variability
        risk_score += features.get('short_interval_risk', 0)
        risk_score += features.get('self_transaction_ratio', 0) > 0.1
        
        features['composite_risk_score'] = risk_score
        
        # 4. 警示指標強度
        alert_indicators = [
            features.get('outbound_amount_risk', 0),
            features.get('high_frequency_risk', 0),
            features.get('short_interval_risk', 0),
            features.get('low_variability_risk', 0),
            features.get('net_outflow', 0)
        ]
        features['alert_indicator_strength'] = sum(alert_indicators)
        
        return features
    
    def calculate_suspicious_pattern_score(self, account_txns):
        """計算可疑模式分數"""
        score = 0
        
        # 1. 自轉交易
        self_txns = account_txns[account_txns['from_acct'] == account_txns['to_acct']]
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
            return 1  # 已知警示帳戶
        elif account in predict_df['acct'].values:
            return -1  # 需要預測的帳戶
        else:
            return 0  # 正常帳戶
    
    def create_zero_alert_features(self):
        """建立零警示特徵"""
        return {
            'outbound_amount_risk': 0,
            'outbound_amount_ratio': 0,
            'high_frequency_risk': 0,
            'transaction_frequency': 0,
            'short_interval_risk': 0,
            'avg_interval_hours': 0,
            'low_variability_risk': 0,
            'amount_cv': 0,
            'net_outflow': 0,
            'outbound_dominance': 0,
            'outbound_counterparty_diversity': 0,
            'inbound_counterparty_diversity': 0,
            'counterparty_imbalance': 0,
            'night_transaction_ratio': 0,
            'night_transaction_count': 0,
            'weekend_transaction_ratio': 0,
            'weekend_transaction_count': 0,
            'interval_variability': 0,
            'interval_std': 0,
            'self_transaction_count': 0,
            'self_transaction_ratio': 0,
            'same_amount_transaction_count': 0,
            'same_amount_transaction_ratio': 0,
            'same_amount_frequency': 0,
            'amount_skewness': 0,
            'amount_kurtosis': 0,
            'amount_concentration': 0,
            'transaction_time_span_hours': 0,
            'transaction_intensity': 0,
            'amount_zscore_max': 0,
            'amount_zscore_mean': 0,
            'amount_outlier_ratio': 0,
            'interval_zscore_max': 0,
            'interval_zscore_mean': 0,
            'suspicious_pattern_score': 0,
            'high_amount_high_frequency': 0,
            'low_variability_high_frequency': 0,
            'composite_risk_score': 0,
            'alert_indicator_strength': 0
        }
