import pandas as pd

full_path  = "/Users/chenzijian/codespace/cits-4404-project/Deliverable02/Data/btc_features.csv"      # 全量特征
train_path = "/Users/chenzijian/codespace/cits-4404-project/Deliverable02/Data/btc_train.csv"     # 你现有的训练CSV
test_path  = "/Users/chenzijian/codespace/cits-4404-project/Deliverable02/Data/btc_test.csv"          # 要保存的测试CSV

# 读全量或直接读 train，然后切 2020-01-01 以后
df = pd.read_csv(full_path, parse_dates=['Date'], index_col='Date')
df = df.sort_index()      
df_train = df[:'2019-12-31']   
df_test = df['2020-01-01':]          # 只要 2020-01-01 及之后
df_test.to_csv(test_path)
df_train.to_csv(train_path)
print("Test file saved to:", test_path, "  rows:", len(df_test))