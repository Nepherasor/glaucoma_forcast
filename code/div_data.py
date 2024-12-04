import pandas as pd

def


try:
    df = pd.read_excel('filtered_data.xlsx', dtype={'id': str})
except Exception as e:
    print(f"读取文件时出错: {e}")

# 确保ID列是字符串类型
df['id'] = df['id'].astype(str)

# 检查读取的前几行数据，确认是否正常
print(df.head())

# 根据ID列分组
grouped = df.groupby('id')

# 遍历分组，并将每个分组保存到单独的CSV文件
for name, group in grouped:
    group.to_excel(f'divdata/patient_{name}.xlsx', index=False)

print("分组完成，每个ID的数据已保存到单独的CSV文件中。")
