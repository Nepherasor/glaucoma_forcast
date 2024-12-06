import pandas as pd
import os
import shutil

def Func_div_id(path):
    try:
        df = pd.read_excel(path, dtype={'id': str})
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

def div_data_md_isNoempty(path):

    # 创建空数据文件夹
    empty_path = os.path.join(path, 'empty')
    os.makedirs(empty_path, exist_ok=True)

    # 创建有数据文件夹
    md_path = os.path.join(path, 'md')
    os.makedirs(md_path, exist_ok=True)

    # 创建至少有4行数据的文件夹
    md4_path = os.path.join(path, 'md4')
    os.makedirs(md4_path, exist_ok=True)

    # 遍历文件夹中的所有Excel文件
    for filename in os.listdir(path):
        # 确保只处理patient_开头的Excel文件
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            # 完整文件路径
            file_path = os.path.join(path, filename)

            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                # 检查第14和15列是否有数据
                col14_data_count = df.iloc[:, 13].dropna().shape[0]
                col15_data_count = df.iloc[:, 14].dropna().shape[0]

                if col14_data_count == 0 and col15_data_count == 0:
                    # 如果两列都完全为空，复制到empty文件夹
                    shutil.copy(file_path, empty_path)
                elif col14_data_count < 4 or col15_data_count < 4:
                    # 如果任一列数据少于4行，复制到md文件夹
                    shutil.copy(file_path, md_path)

                else:
                    # 如果两列至少各有4行数据，复制到md4文件夹
                    shutil.copy(file_path, md4_path)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")


#path = 'filtered_data.xlsx'
div_data_md_isNoempty(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata')