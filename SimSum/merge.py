import pandas as pd

# Load CSV files into DataFrames
df1 = pd.read_csv('/home/tejadhith/Project/NLP/easy-summary/SimSum/data_pg/plaba/plaba_sents_test.csv')
df2 = pd.read_csv('/home/tejadhith/Project/NLP/easy-summary/SimSum/data_pg/plaba/test_output.csv')

# Merge DataFrames on 'pair_id' and 'sent_id'
merged_df = pd.merge(df1, df2, on=['pair_id', 'sent_id','complex'])
print(merged_df.columns)
# print(df1['simple'])
# Create a new column for 'simple' sentences
# merged_df['simple'] = df1['simple']

# Print the resulting DataFrame
print(merged_df.head())
merged_df.to_csv('/home/tejadhith/Project/NLP/easy-summary/SimSum/data_pg/plaba/plaba_output.csv', index=False)
