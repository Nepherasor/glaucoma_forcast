import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Multiply,Masking,Add
import traceback
from tensorflow.keras.optimizers import Adam
import joblib
from tensorflow.keras.models import load_model

features = ['gender', 'age', 'iop', 'cdr', 'period','md','rnfl']
targets = ['md','rnfl']

# 定义数据加载器
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

        self.period_values = None  #周期值
def Func_Dataloader_single(file_path):
    """
    功能：
    加载单个患者的Excel文件，读取其治疗数据，并排序。

    输入：
    - file_path (str): 单个患者数据的Excel文件路径。

    输出：
    - data (Phandle): 一个包含患者信息的 Phandle 对象。
    """
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

    data.period_values = df_sorted.iloc[:, 14]

    return data

def create_sequences(data, time_steps=5):
    """改进的序列生成函数，带动态mask处理"""
    sequences = []
    masks = []
    targets_list = []
    
    for i in range(len(data) - time_steps):
        seq = data.iloc[i:i+time_steps][features].values
        
        # 第一阶段：用相邻值填充紧急缺失（连续缺失不超过2个）
        seq_filled = pd.DataFrame(seq).ffill(limit=2).bfill(limit=2).values
        
        # 生成最终mask（标记原始缺失位置）
        mask = ~np.isnan(seq)  # 记录原始缺失位置
        final_mask = mask.astype(int)
        # final_mask = np.ones_like(seq)
        sequences.append(seq_filled)
        masks.append(final_mask)
        targets_list.append(data.iloc[i+time_steps][targets].values)
    
    return np.nan_to_num(np.array(sequences)), np.array(masks), np.array(targets_list)


def Func_Process_Multi_Inter_LSTM(extend_path,test_path):
    """
    功能：
    使用多输入多输出 LSTM 模型处理数据并预测缺失值。

    输入：
    - extend_path: 训练数据路径
    - test_path: 测试数据路径

    输出：
    - 保存预测结果和模型到指定路径
    """
    # 创建输出文件夹
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
    output_path = r'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str
    os.makedirs(output_path, exist_ok=True)

    # 初始化数据存储
    data_Process_Inter_LSTM_all = []

    # === 数据加载与预处理 ===
    # 遍历训练数据文件
    # 检查模型和标准化参数是否存在
    # model_path = os.path.join('E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code', 'biLSTM_model.h5')
    # scaler_path = os.path.join('E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code', 'scaler.pkl')
    model_path=''
    scaler_path=''
    
            # === 特征工程 ===

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        # === 加载已有模型和参数 ===
        print("加载已有模型和标准化参数...")
        model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        scaler = joblib.load(scaler_path)
        
        
    else:
        # === 需要训练新模型 ===
        print("未找到现有模型，开始训练新模型...")
        # === 数据加载与预处理 ===
        for filename in os.listdir(extend_path):
            if filename.startswith('patient_') and filename.endswith('.xlsx'):
                file_path = os.path.join(extend_path, filename)
                try:
                    data_extend = Func_Dataloader_single(file_path)

                    # 合并左右眼数据
                    eye_data = {
                        'ID': [data_extend.ID * 10 + 1, data_extend.ID * 10 + 2],
                        'gender_values': [data_extend.gender_values, data_extend.gender_values],
                        'age_values': [data_extend.age_values, data_extend.age_values],
                        'iop_values': [data_extend.iop_od_values, data_extend.iop_os_values],
                        'cdr_values': [data_extend.cdr_od_values, data_extend.cdr_os_values],
                        'md_values': [data_extend.md_od_values, data_extend.md_os_values],
                        'rnfl_values': [data_extend.rnfl_od_values, data_extend.rnfl_os_values],
                        'period_values': [data_extend.period_values, data_extend.period_values]
                    }

                    # 转换为DataFrame
                    for i in range(2):
                        df = pd.DataFrame({
                            'ID': eye_data['ID'][i],
                            'gender': eye_data['gender_values'][i],
                            'age': eye_data['age_values'][i],
                            'iop': eye_data['iop_values'][i],
                            'cdr': eye_data['cdr_values'][i],
                            'md': eye_data['md_values'][i],
                            'rnfl': eye_data['rnfl_values'][i],
                            'period': eye_data['period_values'][i]
                        })
                        data_Process_Inter_LSTM_all.append(df)

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
                    traceback.print_exc()

        # 合并所有数据
        LSTM_all_data = pd.concat(data_Process_Inter_LSTM_all, axis=0, ignore_index=True)



        # === 数据标准化 ===
        scaler = StandardScaler()
        scaled_data = LSTM_all_data.copy()
        scaled_data[features] = scaler.fit_transform(scaled_data[features])
        
        scaler_path = os.path.join(output_path, 'scaler.pkl')
        model_path = os.path.join(output_path, 'Multi_LSTM_model.h5')
        joblib.dump(scaler, scaler_path)  # 保存标准化模型



        # 划分数据集
        train_size = int(len(scaled_data) * 0.9)
        val_size = int(len(scaled_data) * 0.1)


        train_data = scaled_data.iloc[:train_size]
        val_data = scaled_data.iloc[train_size:(train_size + val_size)]

        X_train, M_train, y_train = create_sequences(train_data)
        X_val, M_val, y_val = create_sequences(val_data)


        
        # === 模型构建 ===
        def build_model(input_shape):
            """改进的模型结构，带分层mask处理"""
            input_seq = Input(shape=input_shape)
            input_mask = Input(shape=input_shape)
            
            # 动态特征加权
            masked = Multiply()([input_seq, input_mask])
            
            # 分层mask处理
            x = Masking(mask_value=0.)(masked)
            x = Bidirectional(LSTM(128, return_sequences=True))(x)
            x = Dropout(0.3)(x)
            x = Bidirectional(LSTM(64, return_sequences=False))(x)
            
            # 残差连接
            residual = Dense(128, activation='relu')(x)
            x = Add()([x, residual])
            
            output = Dense(len(targets), activation='linear')(x)
            
            return Model(inputs=[input_seq, input_mask], outputs=output)

        # === 模型训练 ===
        model = build_model(input_shape=(5, len(features)))
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse')
        history = model.fit(
            [X_train, M_train], y_train,
            validation_data=([X_val, M_val], y_val),
            epochs=100,
            batch_size=32,
            verbose=2
        )

        # === 保存新模型 ===
        model.save(model_path)
        pd.DataFrame(history.history).to_csv(os.path.join(output_path, 'training_log.csv'))

    # === 加载测试数据 ===
    print("开始处理测试数据...")
    test_data_all = []
    
    # 遍历测试数据文件
    for filename in os.listdir(test_path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path = os.path.join(test_path, filename)
            try:
                data_test = Func_Dataloader_single(file_path)
                
                # 使用与训练数据相同的处理方式合并左右眼数据
                eye_data = {
                    'ID': [data_test.ID * 10 + 1, data_test.ID * 10 + 2],
                    'gender_values': [data_test.gender_values, data_test.gender_values],
                    'age_values': [data_test.age_values, data_test.age_values],
                    'iop_values': [data_test.iop_od_values, data_test.iop_os_values],
                    'cdr_values': [data_test.cdr_od_values, data_test.cdr_os_values],
                    'md_values': [data_test.md_od_values, data_test.md_os_values],
                    'rnfl_values': [data_test.rnfl_od_values, data_test.rnfl_os_values],
                    'period_values': [data_test.period_values, data_test.period_values]
                }

                # 转换为与训练数据相同结构的DataFrame
                for i in range(2):
                    df = pd.DataFrame({
                        'ID': eye_data['ID'][i],
                        'gender': eye_data['gender_values'][i],
                        'age': eye_data['age_values'][i],
                        'iop': eye_data['iop_values'][i],
                        'cdr': eye_data['cdr_values'][i],
                        'md': eye_data['md_values'][i],
                        'rnfl': eye_data['rnfl_values'][i],
                        'period': eye_data['period_values'][i]
                    })
                    test_data_all.append(df)

            except Exception as e:
                print(f"处理测试文件 {filename} 时出错: {e}")
                traceback.print_exc()

    # 合并所有测试数据
    test_data = pd.concat(test_data_all, axis=0, ignore_index=True)
    
    # 使用训练时的scaler进行标准化
    test_data_scaled = test_data.copy()
    test_data_scaled[features] = scaler.transform(test_data_scaled[features])
    
    # 使用相同的create_sequences函数生成测试序列
    X_test, M_test, y_test = create_sequences(test_data_scaled)
    
    # 预测缺失值
    predictions = model.predict([X_test, M_test])
    
    # 逆标准化预测结果
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate([
            X_test[:, -1, :3],  # 保留gender, age, period
            predictions
        ], axis=1)
    )[:, 3:]  # 只取预测的目标列
    
    # 填充缺失值并保存结果
    test_data_filled = test_data.copy()
    for i in range(len(predictions_rescaled)):
        idx = i + 5  # 与create_sequences的time_steps对应
        test_data_filled.loc[idx, targets] = predictions_rescaled[i]
    
    # 保存处理后的测试数据
    test_data_filled.to_csv(os.path.join(output_path, 'test_data_filled.csv'), index=False)
    print("测试数据处理完成，结果已保存至:", output_path)


# 使用示例
extend_LSTM_train_path =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_multi_LSTM_process\train')
extend_LSTM_test_path =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_multi_LSTM_process\test')

Func_Process_Multi_Inter_LSTM(extend_LSTM_train_path,extend_LSTM_test_path)