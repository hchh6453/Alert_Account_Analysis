#!/usr/bin/env python3
"""
特徵工程完整說明文件
詳細解釋每個特徵的意義、計算方法和業務價值
"""

def print_feature_documentation():
    """輸出完整的特徵說明文件"""
    
    print("=" * 80)
    print("金融交易異常檢測系統 - 特徵工程完整說明")
    print("=" * 80)
    
    print("\n📊 總覽")
    print("-" * 40)
    print("本系統從交易資料中提取 42 個特徵，分為 5 大類別：")
    print("1. 時間特徵 (12個) - 分析交易時間模式")
    print("2. 金額特徵 (14個) - 分析交易金額分佈")
    print("3. 交易對手特徵 (7個) - 分析交易關係")
    print("4. 交易模式特徵 (5個) - 檢測異常模式")
    print("5. 網路特徵 (2個) - 分析交易網路結構")
    print("6. 其他特徵 (2個) - 帳戶標識和標籤")
    
    print("\n🕐 一、時間特徵 (Temporal Features)")
    print("-" * 50)
    
    temporal_features = [
        {
            "name": "total_transactions",
            "description": "總交易次數",
            "calculation": "該帳戶的所有交易筆數 (轉入+轉出)",
            "business_value": "反映帳戶活躍度，異常帳戶可能有異常高的交易頻率",
            "anomaly_indicator": "過高或過低都可能異常"
        },
        {
            "name": "transaction_days",
            "description": "交易天數",
            "calculation": "該帳戶有交易記錄的天數",
            "business_value": "反映帳戶使用持續性，異常帳戶可能集中在短時間內大量交易",
            "anomaly_indicator": "交易天數過少但交易量大的帳戶"
        },
        {
            "name": "avg_transactions_per_day",
            "description": "平均每日交易次數",
            "calculation": "total_transactions / transaction_days",
            "business_value": "衡量帳戶日常活躍度，異常帳戶可能有異常高的日交易頻率",
            "anomaly_indicator": "數值異常高"
        },
        {
            "name": "night_transactions",
            "description": "夜間交易次數",
            "calculation": "22:00-06:00 時段的交易次數",
            "business_value": "正常用戶很少在深夜交易，異常帳戶可能利用夜間進行可疑活動",
            "anomaly_indicator": "夜間交易過多"
        },
        {
            "name": "night_transaction_ratio",
            "description": "夜間交易比例",
            "calculation": "night_transactions / total_transactions",
            "business_value": "衡量夜間交易佔比，異常帳戶夜間交易比例通常較高",
            "anomaly_indicator": "比例過高 (>0.3)"
        },
        {
            "name": "weekend_transactions",
            "description": "週末交易次數",
            "calculation": "週六、週日的交易次數",
            "business_value": "正常用戶週末交易較少，異常帳戶可能利用週末進行可疑活動",
            "anomaly_indicator": "週末交易過多"
        },
        {
            "name": "weekend_transaction_ratio",
            "description": "週末交易比例",
            "calculation": "weekend_transactions / total_transactions",
            "business_value": "衡量週末交易佔比，異常帳戶週末交易比例可能異常",
            "anomaly_indicator": "比例過高"
        },
        {
            "name": "transaction_frequency_std",
            "description": "交易頻率標準差",
            "calculation": "每日交易次數的標準差",
            "business_value": "反映交易頻率的穩定性，異常帳戶交易頻率變化劇烈",
            "anomaly_indicator": "標準差過大"
        },
        {
            "name": "transaction_frequency_cv",
            "description": "交易頻率變異係數",
            "calculation": "transaction_frequency_std / 平均每日交易次數",
            "business_value": "標準化的頻率變化指標，異常帳戶變異係數通常較高",
            "anomaly_indicator": "變異係數過高"
        },
        {
            "name": "avg_transaction_interval_hours",
            "description": "平均交易間隔(小時)",
            "calculation": "相鄰交易間的平均時間間隔",
            "business_value": "反映交易節奏，異常帳戶可能有異常短或長的交易間隔",
            "anomaly_indicator": "間隔過短或過長"
        },
        {
            "name": "min_transaction_interval_hours",
            "description": "最小交易間隔(小時)",
            "calculation": "相鄰交易間的最小時間間隔",
            "business_value": "檢測快速連續交易，異常帳戶可能有極短時間間隔的交易",
            "anomaly_indicator": "間隔過短 (<1小時)"
        },
        {
            "name": "transaction_interval_std",
            "description": "交易間隔標準差",
            "calculation": "交易間隔時間的標準差",
            "business_value": "反映交易時間的規律性，異常帳戶間隔時間變化劇烈",
            "anomaly_indicator": "標準差過大"
        }
    ]
    
    for i, feature in enumerate(temporal_features, 1):
        print(f"\n{i:2d}. {feature['name']}")
        print(f"    描述: {feature['description']}")
        print(f"    計算: {feature['calculation']}")
        print(f"    業務價值: {feature['business_value']}")
        print(f"    異常指標: {feature['anomaly_indicator']}")
    
    print("\n💰 二、金額特徵 (Amount Features)")
    print("-" * 50)
    
    amount_features = [
        {
            "name": "total_amount",
            "description": "總交易金額",
            "calculation": "該帳戶所有交易的總金額",
            "business_value": "反映帳戶資金規模，異常帳戶可能有異常大的資金流動",
            "anomaly_indicator": "金額異常大"
        },
        {
            "name": "avg_amount",
            "description": "平均交易金額",
            "calculation": "total_amount / total_transactions",
            "business_value": "反映單筆交易規模，異常帳戶平均金額可能異常",
            "anomaly_indicator": "平均金額過高或過低"
        },
        {
            "name": "max_amount",
            "description": "最大交易金額",
            "calculation": "單筆交易的最大金額",
            "business_value": "檢測大額交易，異常帳戶可能有異常大的單筆交易",
            "anomaly_indicator": "最大金額異常大"
        },
        {
            "name": "min_amount",
            "description": "最小交易金額",
            "calculation": "單筆交易的最小金額",
            "business_value": "檢測小額交易，異常帳戶可能有很多小額測試交易",
            "anomaly_indicator": "最小金額異常小"
        },
        {
            "name": "amount_std",
            "description": "金額標準差",
            "calculation": "交易金額的標準差",
            "business_value": "反映金額分佈的離散程度，異常帳戶金額變化劇烈",
            "anomaly_indicator": "標準差過大"
        },
        {
            "name": "amount_cv",
            "description": "金額變異係數",
            "calculation": "amount_std / avg_amount",
            "business_value": "標準化的金額變化指標，異常帳戶變異係數通常較高",
            "anomaly_indicator": "變異係數過高"
        },
        {
            "name": "large_transaction_count",
            "description": "大額交易數量",
            "calculation": "金額超過95%分位數的交易數量",
            "business_value": "檢測大額交易頻率，異常帳戶大額交易可能過多",
            "anomaly_indicator": "大額交易過多"
        },
        {
            "name": "large_transaction_ratio",
            "description": "大額交易比例",
            "calculation": "large_transaction_count / total_transactions",
            "business_value": "衡量大額交易佔比，異常帳戶大額交易比例可能異常",
            "anomaly_indicator": "比例過高"
        },
        {
            "name": "inbound_amount",
            "description": "轉入金額",
            "calculation": "該帳戶作為轉入方的總金額",
            "business_value": "反映資金流入規模，異常帳戶可能有異常大的資金流入",
            "anomaly_indicator": "轉入金額異常大"
        },
        {
            "name": "outbound_amount",
            "description": "轉出金額",
            "calculation": "該帳戶作為轉出方的總金額",
            "business_value": "反映資金流出規模，異常帳戶可能有異常大的資金流出",
            "anomaly_indicator": "轉出金額異常大"
        },
        {
            "name": "net_amount",
            "description": "淨金額",
            "calculation": "inbound_amount - outbound_amount",
            "business_value": "反映資金淨流入，異常帳戶淨流入可能異常",
            "anomaly_indicator": "淨金額異常大(正或負)"
        },
        {
            "name": "inbound_ratio",
            "description": "轉入比例",
            "calculation": "inbound_amount / total_amount",
            "business_value": "衡量轉入交易佔比，異常帳戶轉入比例可能異常",
            "anomaly_indicator": "比例過高或過低"
        },
        {
            "name": "amount_outlier_count",
            "description": "金額異常值數量",
            "calculation": "使用IQR方法檢測的金額異常值數量",
            "business_value": "檢測金額異常值，異常帳戶可能有較多金額異常值",
            "anomaly_indicator": "異常值過多"
        },
        {
            "name": "amount_outlier_ratio",
            "description": "金額異常值比例",
            "calculation": "amount_outlier_count / total_transactions",
            "business_value": "衡量金額異常值佔比，異常帳戶異常值比例通常較高",
            "anomaly_indicator": "比例過高"
        }
    ]
    
    for i, feature in enumerate(amount_features, 1):
        print(f"\n{i:2d}. {feature['name']}")
        print(f"    描述: {feature['description']}")
        print(f"    計算: {feature['calculation']}")
        print(f"    業務價值: {feature['business_value']}")
        print(f"    異常指標: {feature['anomaly_indicator']}")
    
    print("\n👥 三、交易對手特徵 (Counterparty Features)")
    print("-" * 50)
    
    counterparty_features = [
        {
            "name": "unique_inbound_counterparties",
            "description": "唯一轉入對手數",
            "calculation": "該帳戶作為轉入方的唯一對手帳戶數量",
            "business_value": "反映資金來源多樣性，異常帳戶可能與異常多的對手交易",
            "anomaly_indicator": "對手數過多或過少"
        },
        {
            "name": "unique_outbound_counterparties",
            "description": "唯一轉出對手數",
            "calculation": "該帳戶作為轉出方的唯一對手帳戶數量",
            "business_value": "反映資金去向多樣性，異常帳戶可能與異常多的對手交易",
            "anomaly_indicator": "對手數過多或過少"
        },
        {
            "name": "total_unique_counterparties",
            "description": "總唯一對手數",
            "calculation": "unique_inbound_counterparties + unique_outbound_counterparties",
            "business_value": "反映整體交易關係複雜度，異常帳戶關係可能異常複雜",
            "anomaly_indicator": "總對手數異常"
        },
        {
            "name": "inbound_counterparty_concentration",
            "description": "轉入對手集中度",
            "calculation": "最大轉入對手交易次數 / 總轉入交易次數",
            "business_value": "衡量轉入交易的集中程度，異常帳戶可能過度集中於特定對手",
            "anomaly_indicator": "集中度過高"
        },
        {
            "name": "outbound_counterparty_concentration",
            "description": "轉出對手集中度",
            "calculation": "最大轉出對手交易次數 / 總轉出交易次數",
            "business_value": "衡量轉出交易的集中程度，異常帳戶可能過度集中於特定對手",
            "anomaly_indicator": "集中度過高"
        },
        {
            "name": "repeat_counterparty_ratio",
            "description": "重複對手比例",
            "calculation": "(總交易次數 - 唯一對手數) / 總交易次數",
            "business_value": "衡量與重複對手交易的比例，異常帳戶可能與特定對手重複交易",
            "anomaly_indicator": "重複比例過高"
        },
        {
            "name": "self_transaction_ratio",
            "description": "自轉交易比例",
            "calculation": "自轉交易次數 / 總交易次數",
            "business_value": "檢測自轉交易，異常帳戶可能有異常多的自轉交易",
            "anomaly_indicator": "自轉比例過高"
        }
    ]
    
    for i, feature in enumerate(counterparty_features, 1):
        print(f"\n{i:2d}. {feature['name']}")
        print(f"    描述: {feature['description']}")
        print(f"    計算: {feature['calculation']}")
        print(f"    業務價值: {feature['business_value']}")
        print(f"    異常指標: {feature['anomaly_indicator']}")
    
    print("\n🔍 四、交易模式特徵 (Pattern Features)")
    print("-" * 50)
    
    pattern_features = [
        {
            "name": "burst_transaction_count",
            "description": "爆發性交易數量",
            "calculation": "短時間內異常高頻交易的數量",
            "business_value": "檢測爆發性交易模式，異常帳戶可能有短時間內大量交易",
            "anomaly_indicator": "爆發性交易過多"
        },
        {
            "name": "burst_transaction_ratio",
            "description": "爆發性交易比例",
            "calculation": "burst_transaction_count / total_transactions",
            "business_value": "衡量爆發性交易佔比，異常帳戶爆發性交易比例可能較高",
            "anomaly_indicator": "比例過高"
        },
        {
            "name": "same_amount_transaction_count",
            "description": "相同金額交易數量",
            "calculation": "金額相同的交易數量",
            "business_value": "檢測重複金額交易，異常帳戶可能有大量相同金額的交易",
            "anomaly_indicator": "相同金額交易過多"
        },
        {
            "name": "same_amount_transaction_ratio",
            "description": "相同金額交易比例",
            "calculation": "same_amount_transaction_count / total_transactions",
            "business_value": "衡量相同金額交易佔比，異常帳戶可能重複使用相同金額",
            "anomaly_indicator": "比例過高"
        },
        {
            "name": "regular_time_pattern",
            "description": "規律時間模式",
            "calculation": "特定小時交易次數的最大值 / 總交易次數",
            "business_value": "檢測規律性交易時間，異常帳戶可能在特定時間規律交易",
            "anomaly_indicator": "規律性過強"
        }
    ]
    
    for i, feature in enumerate(pattern_features, 1):
        print(f"\n{i:2d}. {feature['name']}")
        print(f"    描述: {feature['description']}")
        print(f"    計算: {feature['calculation']}")
        print(f"    業務價值: {feature['business_value']}")
        print(f"    異常指標: {feature['anomaly_indicator']}")
    
    print("\n🌐 五、網路特徵 (Network Features)")
    print("-" * 50)
    
    network_features = [
        {
            "name": "transaction_network_density",
            "description": "交易網路密度",
            "calculation": "實際交易連接數 / 最大可能連接數",
            "business_value": "反映交易網路的緊密程度，異常帳戶網路密度可能異常",
            "anomaly_indicator": "密度過高或過低"
        },
        {
            "name": "transaction_network_clustering",
            "description": "交易網路聚類係數",
            "calculation": "轉入和轉出對手重疊數 / 總唯一對手數",
            "business_value": "反映交易網路的聚類程度，異常帳戶聚類係數可能異常",
            "anomaly_indicator": "聚類係數過高"
        }
    ]
    
    for i, feature in enumerate(network_features, 1):
        print(f"\n{i:2d}. {feature['name']}")
        print(f"    描述: {feature['description']}")
        print(f"    計算: {feature['calculation']}")
        print(f"    業務價值: {feature['business_value']}")
        print(f"    異常指標: {feature['anomaly_indicator']}")
    
    print("\n🏷️ 六、其他特徵 (Other Features)")
    print("-" * 50)
    
    other_features = [
        {
            "name": "acct",
            "description": "帳戶ID",
            "calculation": "帳戶的唯一標識符",
            "business_value": "用於識別和追蹤特定帳戶",
            "anomaly_indicator": "N/A"
        },
        {
            "name": "label",
            "description": "帳戶標籤",
            "calculation": "0=正常帳戶, 1=警示帳戶, NaN=無標籤",
            "business_value": "用於監督學習的目標變數",
            "anomaly_indicator": "1表示警示帳戶"
        }
    ]
    
    for i, feature in enumerate(other_features, 1):
        print(f"\n{i:2d}. {feature['name']}")
        print(f"    描述: {feature['description']}")
        print(f"    計算: {feature['calculation']}")
        print(f"    業務價值: {feature['business_value']}")
        print(f"    異常指標: {feature['anomaly_indicator']}")
    
    print("\n" + "=" * 80)
    print("📋 特徵工程總結")
    print("=" * 80)
    
    print("\n🎯 特徵設計原則:")
    print("1. 多維度分析: 從時間、金額、對手、模式、網路等多角度分析")
    print("2. 異常檢測導向: 每個特徵都針對特定的異常模式設計")
    print("3. 業務可解釋性: 特徵具有明確的業務意義")
    print("4. 統計穩健性: 使用標準化指標避免極值影響")
    
    print("\n🔍 異常檢測策略:")
    print("1. 時間異常: 夜間交易、週末交易、爆發性交易")
    print("2. 金額異常: 大額交易、金額異常值、金額分佈異常")
    print("3. 關係異常: 對手集中度、重複交易、自轉交易")
    print("4. 模式異常: 規律性交易、相同金額交易")
    print("5. 網路異常: 網路密度、聚類係數異常")
    
    print("\n📊 特徵重要性:")
    print("1. 高重要性: 交易頻率、金額統計、對手集中度")
    print("2. 中重要性: 時間模式、交易間隔、網路特徵")
    print("3. 輔助性: 規律模式、相同金額、自轉比例")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_feature_documentation()
