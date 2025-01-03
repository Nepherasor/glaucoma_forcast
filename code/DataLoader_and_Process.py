import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import copy

all_missing_data_ratios = []
Sample_Feqs = []
the_last_ages = []


class result_value:
    def __init__(self):
        self.predicted_val = pd.Series()
        self.true_val = pd.Series()

        self.predicted_list = []
        self.true_list = []


class result:
    def __init__(self):
        self.iop_od = result_value()
        self.iop_os = result_value()

        self.cdr_od = result_value()
        self.cdr_os = result_value()

        self.md_od = result_value()
        self.md_os = result_value()

        self.rnfl_od = result_value()
        self.rnfl_os = result_value()


class Phandle:
    def __init__(self):
        # 性别转换：女性为0，男性为1
        self.ID = None
        self.treat_dates = None  # 记录日期
        self.gender_values = None  # 初始化为None

        # 年龄
        self.birth_dates = None  #记录生日
        self.age_values = None

        self.diagnosis_values = None

        # 眼压 (Intraocular Pressure, IOP)
        self.iop_od_values = None  # 右眼眼压
        self.iop_os_values = None  # 左眼眼压

        self.del_iop_od_values = None  # 右眼眼压差
        self.del_iop_os_values = None  # 左眼眼压差

        # 杯盘比 (Cup-to-Disc Ratio, CDR)
        self.cdr_od_values = None  # 右眼杯盘比
        self.cdr_os_values = None  # 左眼杯盘比

        self.del_cdr_od_values = None  # 右眼杯盘比差
        self.del_cdr_os_values = None  # 左眼杯盘比差

        # 视野 (Mean Deviation, MD)
        self.md_od_values = None  # 右眼视野
        self.md_os_values = None  # 左眼视野

        # 视网膜神经纤维层厚度 (Retinal Nerve Fiber Layer, RNFL)
        self.rnfl_od_values = None  # 右眼RNFL
        self.rnfl_os_values = None  # 左眼RNFL

        self.perid_values = None  #周期值


def Func_Dataloader_single(file_path):
    data = Phandle()
    df = pd.read_excel(file_path)
    df_sorted = df.sort_values('treat_dates').reset_index(drop=True)
    data.ID = df_sorted.iloc[:, 0]
    data.gender_values = df_sorted.iloc[:, 1]  # 第十列 性别
    data.birth_dates = df_sorted.iloc[:, 2]
    data.age_values = df_sorted.iloc[:, 3]  # 第十列 年龄

    data.diagnosis_values = df_sorted.iloc[:, 4]
    data.treat_dates = df_sorted.iloc[:, 5]
    data.iop_od_values = df_sorted.iloc[:, 6]  # 第十列 眼压
    data.iop_os_values = df_sorted.iloc[:, 7]  # 第十一列
    # 计算右眼眼压差值

    data.cdr_od_values = df_sorted.iloc[:, 8]  # 第十2列 杯盘比
    data.cdr_os_values = df_sorted.iloc[:, 9]  # 第十3列

    #
    data.md_od_values = df_sorted.iloc[:, 10]  # 第十2列  视野
    data.md_os_values = df_sorted.iloc[:, 11]  # 第十3列

    data.rnfl_od_values = df_sorted.iloc[:, 12]
    data.rnfl_os_values = df_sorted.iloc[:, 13]

    data.perid_values = df_sorted.iloc[:, 14]

    return data

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1))  # 输出一个值，表示预测的iop值
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def Func_Process_Inter_LSTM(data_extend,data_extend_empty,results_Process_Inter_LSTM,output_path):

def Func_Process_Inter_KNN(data_extend,data_extend_empty,results_Process_Inter_KNN,output_path):
    data_Process_Inter_KNN = copy.copy(data_extend)

    data_Process_Inter_KNN.iop_od_values = Func_Algorithm_Inter_KNN(data_extend_empty.iop_od_values)
    data_Process_Inter_KNN.iop_os_values = Func_Algorithm_Inter_KNN(data_extend_empty.iop_os_values)
    # data_Process_Inter_KNN.cdr_od_values = Func_Algorithm_Inter_KNN(data_extend_empty.cdr_od_values)
    # data_Process_Inter_KNN.cdr_os_values = Func_Algorithm_Inter_KNN(data_extend_empty.cdr_os_values)

    mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_empty.iop_od_values,
                                                                         data_Process_Inter_KNN.iop_od_values,
                                                                         data_extend.iop_od_values)
    results_Process_Inter_KNN.iop_od.predicted_list.append(predicted_value)
    results_Process_Inter_KNN.iop_od.true_list.append(true_value)

    mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_empty.iop_os_values,
                                                                         data_Process_Inter_KNN.iop_os_values,
                                                                         data_extend.iop_os_values)
    results_Process_Inter_KNN.iop_os.predicted_list.append(predicted_value)
    results_Process_Inter_KNN.iop_os.true_list.append(true_value)

    path = output_path + '/result_data_extend_KNN_process/'
    Func_output_excel(data_Process_Inter_KNN, path)
def Func_Process_Inter(extend_path, extend_empty_path):
    """
    path (str): 包含患者Excel文件的文件夹路径
    """
    # 创建输出图表的文件夹（如果不存在）

    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str

    results_Process_Inter_KNN = result()
    results_Process_Inter_LSTM = result()

    os.makedirs(output_path, exist_ok=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历文件夹中的所有Excel文件
    for filename in os.listdir(extend_path):
        # 确保只处理patient_开头的Excel文件
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            # 完整文件路径
            file_path1 = os.path.join(extend_path, filename)
            file_path2 = os.path.join(extend_empty_path, filename)
            try:
                # 读取Excel文件
                data_extend = Func_Dataloader_single(file_path1)
                data_extend_empty = Func_Dataloader_single(file_path2)
                # Func_Process_Inter_KNN(data_extend, data_extend_empty, results_Process_Inter_KNN, output_path)

                data_Process_Inter_LSTM = copy.copy(data_extend)
                # data_Process_Inter_LSTM.iop_od_values = Func_Algorithm_Inter_LSTM(data_extend_empty)
                # data_Process_Inter_LSTM.iop_os_values = Func_Algorithm_Inter_LSTM(data_extend_empty)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    # results_Process_Inter_KNN.iop_od.predicted_val = pd.concat(results_Process_Inter_KNN.iop_od.predicted_list,ignore_index=True)
    # results_Process_Inter_KNN.iop_od.true_val = pd.concat(results_Process_Inter_KNN.iop_od.true_list, ignore_index=True)
    # mse,mae,accuracy = Func_Analyse_Evaluate(results_Process_Inter_KNN.iop_od.predicted_val, results_Process_Inter_KNN.iop_od.true_val)
    #
    # # 将 data_extend 转换为 DataFrame
    # df = pd.DataFrame({
    #     'predicted_val': results_Process_Inter_KNN.iop_od.predicted_val,
    #     'true_val': results_Process_Inter_KNN.iop_od.true_val,
    #     'mse':mse,
    #     'mae':mae,
    #     'Acc':accuracy
    #
    # })
    #
    # # 输出成excel文件
    # df.to_excel(output_path + '/Analyse_data_extend_KNN_process' + '.xlsx', index=False, header=True)



def Func_Dataloader2(path):
    """
    遍历文件夹，为每个患者文件绘制图表
    参数:
    path (str): 包含患者Excel文件的文件夹路径
    """
    # 创建输出图表的文件夹（如果不存在）

    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str
    os.makedirs(output_path, exist_ok=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历文件夹中的所有Excel文件
    for filename in os.listdir(path):
        # 确保只处理patient_开头的Excel文件
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            # 完整文件路径
            file_path = os.path.join(path, filename)

            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                phandle = Phandle()
                data_extend = Phandle()
                data_extend_empty = Phandle()

                # phandle.ID = filename
                # data_extend.ID = filename
                # 按照时间排序
                df_sorted = df.sort_values('treat_date').reset_index(drop=True)

                phandle.ID = df_sorted.iloc[:, 0]

                phandle.gender_values = df_sorted.iloc[:, 1]  # 第十列 性别
                phandle.gender_values = df_sorted.iloc[:, 1].apply(lambda x: 0 if x == '女' else 1)
                phandle.birth_dates = df_sorted.iloc[:, 2]
                phandle.age_values = df_sorted.iloc[:, 3]  # 第十列 年龄
                phandle.treat_dates = df_sorted.iloc[:, 4]

                phandle.diagnosis_values = df_sorted.iloc[:, 5]

                phandle.iop_od_values = df_sorted.iloc[:, 9]  # 第十列 眼压
                phandle.iop_os_values = df_sorted.iloc[:, 10]  # 第十一列

                #计算右眼眼压差值
                phandle.del_iop_od_values = [0] + [phandle.iop_od_values[i] - phandle.iop_od_values[i - 1] for i in
                                                   range(1, len(phandle.iop_od_values))]
                # 计算左眼眼压差值
                phandle.del_iop_os_values = [0] + [phandle.iop_os_values[i] - phandle.iop_os_values[i - 1] for i in
                                                   range(1, len(phandle.iop_os_values))]

                phandle.cdr_od_values = df_sorted.iloc[:, 11]  # 第十2列 杯盘比
                phandle.cdr_os_values = df_sorted.iloc[:, 12]  # 第十3列

                # 计算右眼杯盘比的差值，并在开头插入0以保持长度不变
                # 计算右眼杯盘比差值
                phandle.del_cdr_od_values = [0] + [phandle.cdr_od_values[i] - phandle.cdr_od_values[i - 1] for i in
                                                   range(1, len(phandle.cdr_od_values))]

                # 计算左眼杯盘比差值
                phandle.del_cdr_os_values = [0] + [phandle.cdr_os_values[i] - phandle.cdr_os_values[i - 1] for i in
                                                   range(1, len(phandle.cdr_os_values))]
                #
                phandle.md_od_values = df_sorted.iloc[:, 13]  # 第十2列  视野
                phandle.md_os_values = df_sorted.iloc[:, 14]  # 第十3列

                phandle.rnfl_od_values = df_sorted.iloc[:, 20:25].mean(axis=1)
                phandle.rnfl_os_values = df_sorted.iloc[:, 26:31].mean(axis=1)

                data_extend = Func_time_series_extend_nearest(phandle, data_extend)
                data_extend_empty = Func_create_missing_data(data_extend)
                # Func_calculate_empty_data_ratio(phandle,'single')
                #

                # Func_plot(phandle, filename, output_path)
                # 创建Plotly图表


            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    # Func_calculate_empty_data_ratio(0, 'all')


def Func_Algorithm_Inter_KNN(data_serie):
    # 分离出值为666的索引
    data_series = data_serie.copy()

    indices_to_impute = data_series[data_series == 666].index

    # 将值为666的部分替换为 np.nan，以便KNNImputer进行处理
    data_series[data_series == 666] = np.nan

    # 将 Series 转换为 DataFrame，因为 KNNImputer 需要二维数据
    data_df = data_series.to_frame(name='value')

    # 初始化 KNNImputer
    imputer = KNNImputer(n_neighbors=3)

    # 对数据进行填补
    imputed_data = imputer.fit_transform(data_df)

    # 将结果转换回 Series
    imputed_series = pd.Series(imputed_data.flatten(), index=data_series.index)

    # 将填补后的值放回原 Series 中对应值为666的位置
    for index in indices_to_impute:
        data_series.at[index] = imputed_series[index]

    # print(data_series)

    return data_series


def Func_Analyse_Evaluate(predicted_series, true_series, tolerance=0.1):
    # 计算均方误差（MSE）
    mse = mean_squared_error(true_series, predicted_series)

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(true_series, predicted_series)

    # 判断预测值是否在真实值的 ±tolerance 范围内
    is_accurate = ((predicted_series >= true_series * (1 - tolerance)) & (
                predicted_series <= true_series * (1 + tolerance))).astype(int)

    # 全部标记为 1 的真实标签
    true_labels = pd.Series([1] * len(true_series))

    # 计算准确率
    accuracy = accuracy_score(true_labels, is_accurate)

    return mse,mae,accuracy


def Func_Analyse_Evaluate_Series(original_series, filled_series, reference_series):
    """
    对比原始Series中的666部分与填补后的Series，并计算MSE和MAE。

    参数:
    original_series (pd.Series): 原始数据，包含需要填补的666。
    filled_series (pd.Series): 填补后的Series。
    reference_series (pd.Series): 参考Series，包含真实的值，用于对比。

    返回:
    mse (float): 均方误差 (Mean Squared Error)。
    mae (float): 平均绝对误差 (Mean Absolute Error)。
    predicted_values (pd.Series): 填补后的预测值。
    true_values (pd.Series): 真实的参考值。
    """
    # 只计算原本是666的部分（即NaN的位置）
    mask = original_series == 666  # 用666替代NaN的位置

    # 获取填补后的预测值（只关注填补的部分）
    predicted_values = filled_series[mask]

    # 获取参考数据中的真实值（即原本为666的部分）
    true_values = reference_series[mask]

    # 计算均方误差 (MSE) 和平均绝对误差 (MAE)
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)

    # print(mse,mae,predicted_values, true_values)
    # 返回误差指标
    return mse, mae, predicted_values, true_values


def Func_origin_data_feq(data):
    time_scale = round(
        np.sum([(data.treat_dates[i] - data.treat_dates[i - 1]).days for i in range(1, len(data.treat_dates))]) / len(
            data.treat_dates))
    the_last_age = data.age_values[len(data.age_values) - 1]

    if time_scale > 180:
        time_scale = 180
    elif time_scale < 14:
        time_scale = 14

    # the_first_age = data.age_values[0]
    return time_scale, the_last_age


def Func_time_series_extend_nearest(data, data_extend):
    Sample_Feq, the_last_age = Func_origin_data_feq(data)
    # # Sample_Feqs.append(Sample_Feq)
    # # the_last_ages.append(the_last_age)
    #
    data_extend = Func_expand_time_series_with_neighbors(data, data_extend, Sample_Feq)
    return data_extend


def Func_expand_time_series_with_neighbors(data, data_extend, Sample_Feq):
    """
    扩展周期不固定的时序数据为固定周期，保留有效数据数，其他时间点为空。

    :param original_data: 原始数据，包含时间戳和相应的值
    :param fixed_period_length: 固定周期的长度（扩展后的周期长度）
    :return: 固定周期的数据
    """
    # 假设original_data是一个DataFrame，包含 'time' 和 'value' 列

    # 获取原始时间的起止时间
    start_time = data.treat_dates[0]
    end_time = data.treat_dates[len(data.treat_dates) - 1]

    # 生成固定周期长度的时间点
    expanded_time = pd.date_range(start=start_time, end=end_time, freq=pd.Timedelta(days=Sample_Feq))

    data.perid_values = pd.Series(Sample_Feq)
    # data_extend.perid_values = Sample_Feq

    for attribute in ['ID', 'gender_values', 'birth_dates', 'perid_values']:
        values = getattr(data, attribute)
        # 存储扩展后的值
        expanded_values = []
        # 获取原始时间的时间戳
        # original_times = data.treat_dates
        for time_point in expanded_time:
            expanded_values.append(values.iloc[0])
        setattr(data_extend, attribute, pd.Series(expanded_values))

    # 对每个属性进行扩展
    for attribute in ['age_values', 'diagnosis_values', 'iop_od_values', 'iop_os_values',
                      'cdr_od_values', 'cdr_os_values', 'md_od_values', 'md_os_values',
                      'rnfl_od_values', 'rnfl_os_values']:

        # 获取原始数据的值
        values = getattr(data, attribute)

        # 存储扩展后的值
        expanded_values = []

        # 获取原始时间的时间戳
        original_times = data.treat_dates

        for time_point in expanded_time:
            # 找到最接近的原始时间点
            time_diff = np.abs(original_times - time_point)
            closest_idx = time_diff.argmin()  # 获取最接近的时间点的索引

            # 计算时间差
            if (time_diff[closest_idx] > pd.Timedelta(days=Sample_Feq)):
                # print(time_diff[closest_idx],pd.Timedelta(Sample_Feq))
                expanded_values.append(np.nan)  # 或者使用 None，填充为 NaN
            else:
                # 获取最接近原始时间点的值
                expanded_values.append(values.iloc[closest_idx])
        # 将扩展后的时间戳存储

        # 将扩展后的值赋给 data_extend 对象
        setattr(data_extend, attribute, pd.Series(expanded_values))

    data_extend.treat_dates = pd.Series(expanded_time)
    if(len(data_extend.treat_dates)>2):
        path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_process'
    ####输出csv
        Func_output_excel(data_extend, path)

    return data_extend



def Func_output_excel(data_extend,path):
    ####输出excel
    # 将 data_extend 转换为 DataFrame
    df = pd.DataFrame({
        'id': data_extend.ID,
        'gender_values': data_extend.gender_values,
        'birth_dates': data_extend.birth_dates,
        'age_values': data_extend.age_values,
        'diagnosis_values': data_extend.diagnosis_values,
        'treat_dates': data_extend.treat_dates,

        'iop_od_values': data_extend.iop_od_values,
        'iop_os_values': data_extend.iop_os_values,
        'cdr_od_values': data_extend.cdr_od_values,
        'cdr_os_values': data_extend.cdr_os_values,
        'md_od_values': data_extend.md_od_values,
        'md_os_values': data_extend.md_os_values,
        'rnfl_od_values': data_extend.rnfl_od_values,
        'rnfl_os_values': data_extend.rnfl_os_values,
        'period_values': data_extend.perid_values
    })

    # 输出成excel文件

    os.makedirs(path, exist_ok=True)
    df.to_excel(path + '/patient_' + str(data_extend.ID[0]) + '.xlsx', index=False, header=True)

#眼压 0.15 杯盘比 0.3 视野 0.5 RNFL 0.5
def Func_create_missing_data(data):
    data_empty = data
    for attribute in ['iop_od_values', 'iop_os_values', 'cdr_od_values', 'cdr_os_values',
                      'md_od_values', 'md_os_values', 'rnfl_od_values', 'rnfl_os_values']:

        # 获取原始数据的值
        values = getattr(data, attribute)
        if 'iop' in attribute:
            setattr(data_empty, attribute, Func_cmd_1(values, 0.15))
        # elif 'cdr' in attribute:
        #     setattr(data_empty, attribute, Func_cmd_1(values, 0.3))
        # elif 'md' in attribute:
        #     setattr(data_empty, attribute, Func_cmd_1(values, 0.15))
        # elif 'rnfl' in attribute:
        #     setattr(data_empty, attribute, Func_cmd_1(values, 0.3))

    # 将 data_extend 转换为 DataFrame

    if (len(data_empty.treat_dates)>2):
        path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_empty_process'
        Func_output_excel(data_empty, path)
    return data_empty


def Func_cmd_1(series, missing_rate):
    """
    找出Series的非空数据，并按照给定概率制造缺失数据，缺失部分用X代替。

    :param series: 输入的Series
    :param missing_rate: 缺失数据的概率，取值范围为0到1
    :return: 处理后的Series
    """
    # 找出非空数据
    non_missing_series = series.dropna()

    # 计算需要制造缺失数据的数量
    num_missing = int(np.ceil(len(non_missing_series) * missing_rate))

    # 随机选择需要制造缺失数据的索引
    missing_indices = np.random.choice(non_missing_series.index, num_missing, replace=False)

    # 复制原始Series
    new_series = series.copy()

    # 将选择的索引位置的数据替换为X
    new_series.loc[missing_indices] = 666

    return new_series


def Func_calculate_empty_data_ratio(data, flag):
    """
    计算单个患者各项测量值的空数据占比。

    参数:
    phandle -- 一个对象，包含患者的测量值数据，如年龄、眼压、杯盘比等。

    返回:
    dict -- 包含每项测量空数据占比的字典。
    """
    missing_data_ratios = {}
    average_missing_data_ratios = {}
    # 定义测量值列名和对应的描述
    measurements = {
        'iop_od_values': '眼压(右眼)',
        'iop_os_values': '眼压(左眼)',
        'cdr_od_values': '杯盘比(右眼)',
        'cdr_os_values': '杯盘比(左眼)',
        'md_od_values': '视野(右眼)',
        'md_os_values': '视野(左眼)',
        'rnfl_od_values': 'RNFL(右眼)',
        'rnfl_os_values': 'RNFL(左眼)'
    }
    if flag == 'single':
        # 计算每项测量的空数据占比
        for key, description in measurements.items():
            if hasattr(data, key):  # 检查属性是否存在
                total_values = len(data.__getattribute__(key))
                missing_values = data.__getattribute__(key).isnull().sum() if isinstance(data.__getattribute__(key),
                                                                                         pd.Series) else data.__getattribute__(
                    key).isna().sum()
                missing_data_ratios[description] = missing_values / total_values if total_values > 0 else 0.0

        all_missing_data_ratios.append(missing_data_ratios)
        print(missing_data_ratios)
        return missing_data_ratios

    elif flag == 'all':
        for description in all_missing_data_ratios[0].keys():  # 假设所有字典都有相同的键
            average_missing_data_ratios[description] = sum(
                ratio[description] for ratio in all_missing_data_ratios) / len(all_missing_data_ratios)

        # 输出每个类型的空数据占比的平均值
        print('总平均：', average_missing_data_ratios)
        return average_missing_data_ratios


def Func_Summary_Feq_plot(output_path, Sample_Feqs, the_last_ages):
    plot_frame = pd.DataFrame({
        'Sample_Feq': Sample_Feqs,
        'the_last_age': the_last_ages
    })

    # 创建一个新的列，标记每个用户的年龄组
    bins = [0, 14, 34, 64, np.inf]
    labels = ['0-14', '15-34', '35-64', '>64']
    plot_frame['age_group'] = pd.cut(plot_frame['the_last_age'], bins=bins, labels=labels)

    # 创建子图列表
    figs = []

    # 分组绘制每个年龄段的 Sample_Feq 分布
    for age_group in labels:
        # 获取当前年龄组的数据
        group_data = plot_frame[plot_frame['age_group'] == age_group]

        # 创建Histogram
        hist = px.histogram(group_data, x='Sample_Feq', nbins=30,
                            title=f'Sample_Feq Distribution for Age Group {age_group}')

        figs.append(hist)

    # 计算每个年龄段的平均值或中位数
    age_group_stats = plot_frame.groupby('age_group')['Sample_Feq'].agg(['mean', 'median']).reset_index()

    # 绘制汇总图：显示四个年龄段的平均值或中位数
    summary_fig = px.bar(age_group_stats, x='age_group', y='mean',
                         title='Average Sample_Feq by Age Group', labels={'mean': 'Average Sample_Feq'})
    figs.append(summary_fig)

    # 绘制第五张图：每个用户的 Sample_Feq 与 their last age 的关系
    age_vs_feq_fig = px.scatter(plot_frame, x='the_last_age', y='Sample_Feq',
                                title='Sample_Feq vs Last Age (the_last_age)',
                                labels={'the_last_age': 'Age', 'Sample_Feq': 'Sample_Feq'})
    figs.append(age_vs_feq_fig)

    # 将所有图表保存为一个HTML文件
    output_file = os.path.join(output_path, "All_Sample_Feq_Distributions_and_Summary.html")
    # 将每个图表保存到同一个HTML文件中
    with open(output_file, 'w') as f:
        for fig in figs:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    print(f"All charts saved to {output_file}")


def Func_plot(data, filename, output_path):
    # 创建包含六个子图的Figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '眼压变化', '眼压差变化', '杯盘比变化', '杯盘比差变化',
            '视野变化', 'RNFL变化'
        )
    )

    # 定义颜色
    colors = {
        '右眼眼压': 'blue',
        '左眼眼压': 'red',
        '右眼眼压差变化': 'navy',
        '左眼眼压差变化': 'darkred',
        '右眼杯盘比变化': 'green',
        '左眼杯盘比变化': 'orange',
        '右眼杯盘比差变化': 'forestgreen',
        '左眼杯盘比差变化': 'darkorange',
        '右眼视野': 'purple',
        '左眼视野': 'brown',
        '右眼RNFL': 'pink',
        '左眼RNFL': 'cyan'
    }

    # 眼压数据
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.iop_od_values,
        mode='markers+lines',
        name='右眼眼压',
        marker=dict(color=colors['右眼眼压'], size=10)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.iop_os_values,
        mode='markers+lines',
        name='左眼眼压',
        marker=dict(color=colors['左眼眼压'], size=10)
    ), row=1, col=1)

    # 眼压差数据
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.del_iop_od_values,
        mode='markers+lines',
        name='右眼眼压差变化',
        marker=dict(color=colors['右眼眼压差变化'], size=10)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.del_iop_os_values,
        mode='markers+lines',
        name='左眼眼压差变化',
        marker=dict(color=colors['左眼眼压差变化'], size=10)
    ), row=2, col=1)

    # 杯盘比数据
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.cdr_od_values,
        mode='markers+lines',
        name='右眼杯盘比变化',
        marker=dict(color=colors['右眼杯盘比变化'], size=10)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.cdr_os_values,
        mode='markers+lines',
        name='左眼杯盘比变化',
        marker=dict(color=colors['左眼杯盘比变化'], size=10)
    ), row=1, col=2)

    # 杯盘比差数据
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.del_cdr_od_values,
        mode='markers+lines',
        name='右眼杯盘比差变化',
        marker=dict(color=colors['右眼杯盘比差变化'], size=10)
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.del_cdr_os_values,
        mode='markers+lines',
        name='左眼杯盘比差变化',
        marker=dict(color=colors['左眼杯盘比差变化'], size=10)
    ), row=2, col=2)

    # 视野数据
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.md_od_values,
        mode='markers+lines',
        name='右眼视野',
        marker=dict(color=colors['右眼视野'], size=10)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.md_os_values,
        mode='markers+lines',
        name='左眼视野',
        marker=dict(color=colors['左眼视野'], size=10)
    ), row=3, col=1)

    # RNFL数据
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.rnfl_od_values,
        mode='markers+lines',
        name='右眼RNFL',
        marker=dict(color=colors['右眼RNFL'], size=10)
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=data.age_values,
        y=data.rnfl_os_values,
        mode='markers+lines',
        name='左眼RNFL',
        marker=dict(color=colors['左眼RNFL'], size=10)
    ), row=3, col=2)

    # 更新布局
    fig.update_layout(
        height=1200,  # 增加高度以适应更多行
        width=1600,
        title_text=f'患者 {filename.replace("patient_", "").replace(".xlsx", "")} 的治疗数据',
        template='plotly_white',
        showlegend=True
    )

    # 更新x、y轴标签
    fig.update_xaxes(title_text='年龄', row=1, col=1)
    fig.update_yaxes(title_text='眼压值', row=1, col=1)
    fig.update_xaxes(title_text='年龄', row=2, col=1)
    fig.update_yaxes(title_text='眼压差值', row=2, col=1)
    fig.update_xaxes(title_text='年龄', row=3, col=1)
    fig.update_yaxes(title_text='视野指标值', row=3, col=1)

    fig.update_xaxes(title_text='年龄', row=1, col=2)
    fig.update_yaxes(title_text='杯盘比值', row=1, col=2)
    fig.update_xaxes(title_text='年龄', row=2, col=2)
    fig.update_yaxes(title_text='杯盘比差值', row=2, col=2)
    fig.update_xaxes(title_text='年龄', row=3, col=2)
    fig.update_yaxes(title_text='RNFL厚度值', row=3, col=2)

    # 保存交互式HTML文件
    output_file = os.path.join(output_path, f'{filename.replace(".xlsx", ".html")}')
    pyo.plot(fig, filename=output_file, auto_open=False)
    print(f"成功为 {filename} 生成图表: {output_file}")


# 主执行部分
if __name__ == '__main__':
    # 指定文件夹路径
    # folder_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data')
    extend_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_process')
    extend_empty_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_empty_process')
    # 加载数据并绘制图表
    # Func_Dataloader2(folder_path)          #制造缺失数据和对比
    Func_Process_Inter(extend_path, extend_empty_path)
    # time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    # output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str
    # os.makedirs(output_path, exist_ok=True)
    # Func_Summary_Feq_plot(output_path,Sample_Feqs, the_last_ages)
