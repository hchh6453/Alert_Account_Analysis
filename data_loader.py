"""
資料載入模組
負責載入和初步處理交易資料、警示帳戶資料和預測目標資料
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    """資料載入器"""
    
    def __init__(self):
        self.data_path = "data/"
    
    def load_data(self):
        """載入所有資料"""
        print("載入交易資料...")
        transactions_df = self.load_transactions()
        
        print("載入警示帳戶資料...")
        alerts_df = self.load_alerts()
        
        print("載入預測目標資料...")
        predict_df = self.load_predict()
        
        return transactions_df, alerts_df, predict_df
    
    def load_transactions(self):
        """載入交易資料"""
        df = pd.read_csv(f"{self.data_path}acct_transaction.csv")
        
        # 資料清理和轉換
        df = self.clean_transaction_data(df)
        
        return df
    
    def load_alerts(self):
        """載入警示帳戶資料"""
        df = pd.read_csv(f"{self.data_path}acct_alert.csv")
        
        # 添加標籤
        df['label'] = 1  # 警示帳戶標籤為1
        
        return df
    
    def load_predict(self):
        """載入預測目標資料"""
        df = pd.read_csv(f"{self.data_path}acct_predict.csv")
        
        return df
    
    def clean_transaction_data(self, df):
        """清理交易資料"""
        print("清理交易資料...")
        
        # 轉換資料類型
        df['txn_amt'] = pd.to_numeric(df['txn_amt'], errors='coerce')
        df['txn_date'] = pd.to_numeric(df['txn_date'], errors='coerce')
        
        # 處理時間欄位
        df['txn_datetime'] = self.create_datetime(df['txn_date'], df['txn_time'])
        
        # 移除無效資料
        df = df.dropna(subset=['txn_amt', 'txn_date'])
        
        # 移除異常值
        df = df[(df['txn_amt'] > 0) & (df['txn_amt'] < 1e8)]  # 金額範圍限制
        
        print(f"清理後交易資料: {len(df):,} 筆")
        
        return df
    
    def create_datetime(self, dates, times):
        """建立完整的日期時間"""
        # 假設日期是從某個基準日期開始的天數
        base_date = datetime(2023, 1, 1)  # 假設基準日期
        
        datetimes = []
        for date, time in zip(dates, times):
            try:
                # 計算實際日期
                actual_date = base_date + timedelta(days=int(date))
                
                # 解析時間
                hour, minute, second = map(int, time.split(':'))
                actual_datetime = actual_date.replace(hour=hour, minute=minute, second=second)
                
                datetimes.append(actual_datetime)
            except:
                datetimes.append(None)
        
        return pd.to_datetime(datetimes)
    
    def get_account_list(self, transactions_df, alerts_df, predict_df):
        """取得所有帳戶列表"""
        # 從交易資料中取得所有帳戶
        from_accounts = set(transactions_df['from_acct'].unique())
        to_accounts = set(transactions_df['to_acct'].unique())
        all_accounts = from_accounts.union(to_accounts)
        
        # 從警示和預測資料中取得帳戶
        alert_accounts = set(alerts_df['acct'].unique())
        predict_accounts = set(predict_df['acct'].unique())
        
        # 合併所有帳戶
        all_accounts = all_accounts.union(alert_accounts).union(predict_accounts)
        
        return list(all_accounts)
