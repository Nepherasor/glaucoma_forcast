import pandas as pd
import os
import shutil

import random


def Func_shuffle_and_divide_files(src_path, train_path, vaild_path, test_path, train_num=460, vaild_num=197):
    # 确保目标文件夹存在
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(vaild_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # 获取源文件夹中所有文件的列表
    files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]

    # 打乱文件顺序
    random.shuffle(files)

    # 选择文件复制到文件夹A
    for f in files[:train_num]:
        shutil.copy(os.path.join(src_path, f), os.path.join(train_path, f))

    # 选择文件复制到文件夹B
    for f in files[train_num:train_num + vaild_num]:
        shutil.copy(os.path.join(src_path, f), os.path.join(vaild_path, f))

    # 剩下的文件复制到文件夹C
    for f in files[train_num + vaild_num:]:
        shutil.copy(os.path.join(src_path, f), os.path.join(test_path, f))



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

def check_columns_and_move(file_path, columns, dest_paths, condition):
    try:
        df = pd.read_excel(file_path)
        for col_idx in columns:
            if not df.iloc[:, col_idx].notnull().all():
                shutil.copy(file_path, dest_paths['empty'])
                return
        if condition(df, columns):
            shutil.copy(file_path, dest_paths['full'])
        else:
            shutil.copy(file_path, dest_paths['empty'])
    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")


import os
import shutil
import pandas as pd

def check_columns_and_move_vaild(file_path, columns, dest_paths, condition):
    try:
        df = pd.read_excel(file_path)

        # 如果选中的列中有任何一列完全为空，直接移动到 empty 文件夹
        for col_idx in columns:
            if df.iloc[:, col_idx].dropna().empty:  # 如果该列所有行都没有数据
                shutil.copy(file_path, dest_paths['empty'])  # 移动到 empty 文件夹
                return  # 一旦发现有列为空，直接返回，不再做后续检查

        # 如果所有选中的列都有数据，应用 condition 函数进行进一步验证
        if condition(df, columns):
            shutil.copy(file_path, dest_paths['full'])  # 移动到 valid 文件夹
        else:
            shutil.copy(file_path, dest_paths['empty'])  # 如果不符合 condition 条件，移动到 empty 文件夹

    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")


def div_data_isNoempty(path, columns, full_path_suffix, empty_path_suffix):
    # 创建文件夹
    full_path = os.path.join(path+'/../', f'{full_path_suffix}')
    empty_path = os.path.join(path+'/../', f'{empty_path_suffix}')
    os.makedirs(full_path, exist_ok=True)
    os.makedirs(empty_path, exist_ok=True)
    print(full_path)
    # 遍历文件夹中的所有Excel文件
    for filename in os.listdir(path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path = os.path.join(path, filename)
            check_columns_and_move_vaild(file_path, columns, {'full': full_path, 'empty': empty_path}, lambda df, cols: all(df.iloc[:, col].shape[0]==df.iloc[:, 1].shape[0] and df.iloc[:, col].dropna().shape[0] > 1 for col in cols))




def div_data_isVaild(path, columns, full_path_suffix, empty_path_suffix):
    # 创建文件夹
    full_path = os.path.join(path, f'{full_path_suffix}')
    empty_path = os.path.join(path, f'{empty_path_suffix}')
    os.makedirs(full_path, exist_ok=True)
    os.makedirs(empty_path, exist_ok=True)

    # 遍历文件夹中的所有Excel文件
    for filename in os.listdir(path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path = os.path.join(path, filename)
            check_columns_and_move(file_path, columns, {'full': full_path, 'empty': empty_path}, lambda df, cols: all(df.iloc[:, col].dropna().shape[0] > 2 for col in cols))

# path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata'

# path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data'
# div_data_isNoempty(
#     path=path,
#     columns=[9, 10],  # IOP columns
#     full_path_suffix='iop',
#     empty_path_suffix='iop_empty'
# )
#
# div_data_isNoempty(
#     path=path,
#     columns=[11, 12],  # Ratio columns
#     full_path_suffix='ratio',
#     empty_path_suffix='ratio_empty'
# )
#
# div_data_isNoempty(
#     path=path,
#     columns=[9, 10, 11, 12, 20, 21, 26, 27],  # All columns
#     full_path_suffix='all4',
#     empty_path_suffix='all4_empty'
# )
#


div_data_isNoempty(
    path=r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\fill_data',
    columns=[10,11],  # IOP and Ratio columns
    full_path_suffix='vaild_md_data',
    empty_path_suffix='vaild_md_data_empty'
)

# div_data_isVaild(
#     path=r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\fill_data',
#     columns=[10,11],  # all columns
#     full_path_suffix='vaild_md_data',
#     empty_path_suffix='vaild_md_data_empty'
# )

# div_data_iopandratio_isNoempty(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata')

# Func_shuffle_and_divide_files(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\data\vaild_data', r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\data\datageneration_data\train', r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\data\datageneration_data\vaild', r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\data\datageneration_data\test',)
# Func_shuffle_and_divide_files(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_process',r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\train',r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\val',r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\test')