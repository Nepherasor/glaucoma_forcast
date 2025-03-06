import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
import numpy as np
import tensorflow as tf
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense,Masking,Dropout,TimeDistributed,Concatenate
from tensorflow.keras import layers, Model, Input, Sequential
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import traceback
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
import copy
import gc
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 存储所有缺失数据比例
all_missing_data_ratios = []
# 存储采样频率
Sample_Feqs = []
# 存储最后年龄
the_last_ages = []


class result_value:
    """
    存储预测值和真实值的类
    """
    def __init__(self):
        self.predicted_val = pd.Series()
        self.true_val = pd.Series()

        self.predicted_list = []
        self.true_list = []


class result:
    """
    存储各项指标预测结果的类
    """
    def __init__(self):
        self.iop_od = result_value() # 右眼眼压
        self.iop_os = result_value() # 左眼眼压
        
        self.cdr_od = result_value() # 右眼杯盘比
        self.cdr_os = result_value() # 左眼杯盘比
        
        self.md_od = result_value() # 右眼视野
        self.md_os = result_value() # 左眼视野
        
        self.rnfl_od = result_value() # 右眼视网膜神经纤维层厚度
        self.rnfl_os = result_value() # 左眼视网膜神经纤维层厚度


class Phandle:
    """
    存储患者数据的类
    """
    def __init__(self):
        self.ID = None # 患者ID
        self.treat_dates = None # 就诊日期
        self.gender_values = None # 性别(0-女,1-男)
        
        self.birth_dates = None # 出生日期
        self.age_values = None # 年龄
        
        self.diagnosis_values = None # 诊断结果
        
        # 眼压数据
        self.iop_od_values = None # 右眼眼压
        self.iop_os_values = None # 左眼眼压
        
        self.del_iop_od_values = None # 右眼眼压差值
        self.del_iop_os_values = None # 左眼眼压差值
        
        # 杯盘比数据
        self.cdr_od_values = None # 右眼杯盘比
        self.cdr_os_values = None # 左眼杯盘比
        
        self.del_cdr_od_values = None # 右眼杯盘比差值
        self.del_cdr_os_values = None # 左眼杯盘比差值
        
        # 视野数据
        self.md_od_values = None # 右眼视野
        self.md_os_values = None # 左眼视野
        
        # 视网膜神经纤维层厚度数据
        self.rnfl_od_values = None # 右眼RNFL
        self.rnfl_os_values = None # 左眼RNFL
        
        self.period_values = None # 就诊周期


def Func_Dataloader_single(file_path):
    """
    加载单个患者的Excel文件数据
    
    Args:
        file_path: Excel文件路径
        
    Returns:
        data: 包含患者数据的Phandle对象
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



# def Func_build_gan_model(input_dim):
#     """
#     构建GAN模型
    
#     Args:
#         input_dim: 输入维度
        
#     Returns:
#         generator: 生成器模型
#         discriminator: 判别器模型
#         gan: 完整GAN模型
#     """
#     # 构建生成器
#     generator = Sequential([
#         Dense(256, input_dim=input_dim, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
#         BatchNormalization(momentum=0.8),
#         LeakyReLU(alpha=0.2),
#         Dropout(0.2),
#         Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
#         BatchNormalization(momentum=0.8),
#         LeakyReLU(alpha=0.2),
#         Dropout(0.2),
#         Dense(input_dim, activation='tanh')  # 输出与输入维度一致
#     ], name="Generator")

#     # 构建判别器
#     discriminator = Sequential([
#         Dense(512, input_dim=input_dim, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
#         LeakyReLU(alpha=0.2),
#         Dropout(0.3),  # 防止过拟合
#         Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
#         LeakyReLU(alpha=0.2),
#         Dropout(0.3),  # 防止过拟合
#         Dense(1, activation='sigmoid')  # 二分类输出
#     ], name="Discriminator")
#     discriminator.compile(
#         optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     # 构建完整GAN
#     discriminator.trainable = False
#     gan = Sequential([generator, discriminator], name="GAN")
#     gan.compile(
#         optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
#         loss='binary_crossentropy'
#     )

#     return generator, discriminator, gan


# def Func_Process_Inter_GAN_Multimodal(extend_train_path,extend_test_path, extend_val_path, gan_model_path=None):
#     """
#     使用多模态GAN模型处理数据
    
#     Args:
#         extend_train_path: 训练数据路径
#         extend_test_path: 测试数据路径
#         extend_val_path: 验证数据路径
#         gan_model_path: 预训练GAN模型路径(可选)
        
#     Returns:
#         None,结果保存到文件
#     """
#     time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
#     output_path = f'E:/BaiduSyncdisk/QZZ/data_generation/output/GAN_Multimodal_{time_str}'
#     os.makedirs(output_path, exist_ok=True)

#     # 创建输出文件夹
#     feature_columns = ['age_values','iop_od_values','iop_os_values','cdr_od_values','cdr_os_values','md_od_values','md_os_values']

#     # gan_model_path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code\GAN_inter_model.keras'
#     results_Process_Inter_GAN = result()
#     # 构建或加载 GAN 模型
#     input_dim = len(feature_columns)  # 多模态特征数量：IOP, CDR, MD
#     scaler = StandardScaler()
#     if gan_model_path:
#         generator = load_model(gan_model_path)
#     else:
#         generator, discriminator, gan = Func_build_gan_model(input_dim)

#         data_Process_Inter_GAN_all = []
#         #训练部分
#         # 遍历文件夹中的所有Excel文件
#         for filename in os.listdir(extend_train_path):
#             if filename.startswith('patient_') and filename.endswith('.xlsx'):
#                 file_path = os.path.join(extend_train_path, filename)
#                 try:
#                     # 读取 Excel 文件
#                     data_extend = Func_Dataloader_single(file_path)

#                     # 提取多模态特征
#                     data_Process_Inter_GAN_single = {}
#                     for feature in feature_columns:
#                         data_Process_Inter_GAN_single[feature] = data_extend.__getattribute__(feature)

#                     data_Process_Inter_GAN_all.append(pd.DataFrame(data_Process_Inter_GAN_single))

#                 except Exception as e:
#                     print(f"处理文件 {filename} 时出错: {e}")

#         # 合并所有数据
#         GAN_all_data = pd.concat(data_Process_Inter_GAN_all, axis=0, ignore_index=True)

#         # 缺失值掩码
#         missing_mask = GAN_all_data.isna()

#         # 填充 NaN 值为 0（仅用于训练 GANs，不修改原数据）
#         combined_data = GAN_all_data.fillna(0)

#         # 标准化数据

#         GAN_all_data_normalized = pd.DataFrame(
#             scaler.fit_transform(combined_data),
#             columns=feature_columns
#         )

#         # 转为 DataFrame
#         GAN_all_data_normalized = pd.DataFrame(GAN_all_data_normalized, columns=feature_columns)


#         # 训练 GANs
#         generator = Func_train_gan_multimodal(generator, discriminator, gan, GAN_all_data_normalized, missing_mask,input_dim)

#         generator.save(output_path +'\GAN_inter_model.keras') # 默认保存模型和权重

#     # ##预测部分

#     for filename in os.listdir(extend_test_path):
#         if filename.startswith('patient_') and filename.endswith('.xlsx'):
#             file_path_test = os.path.join(extend_test_path, filename)
#             file_path_val = os.path.join(extend_val_path, filename)
#             try:
#                 # 加载测试和验证数据
#                 data_extend_test = Func_Dataloader_single(file_path_test)
#                 data_extend_test_ori = copy.copy(data_extend_test)
#                 data_extend_val = Func_Dataloader_single(file_path_val)


#                 ###缺失部分
#                 test_column = 'iop_os_values'

#                 # 提取多模态特征
#                 data_Process_Inter_GAN_single = {}
#                 for feature in feature_columns:
#                     data_Process_Inter_GAN_single[feature] = data_extend_test.__getattribute__(feature).replace(666,
#                                                                                                                 np.nan)

#                 # 转换为 DataFrame
#                 test_data = pd.DataFrame(data_Process_Inter_GAN_single)

#                 # 标准化测试数据
#                 test_data_normalized = pd.DataFrame(
#                     scaler.fit_transform(test_data),
#                     columns=feature_columns
#                 )

#                 # 生成缺失值掩码
#                 missing_mask = test_data.isna()

#                 # 用生成器生成缺失值
#                 noise = np.random.normal(0, 1, size=(missing_mask[test_column].sum(), len(feature_columns)))
#                 generated_values = generator.predict(noise)

#                 # 检查生成器输出的形状
#                 assert generated_values.shape[0] == missing_mask[test_column].sum(), \
#                     f"生成器生成的数据数量 ({generated_values.shape[0]}) 与缺失值数量 ({missing_mask[test_column].sum()}) 不一致"

#                 # 填补缺失值
#                 filled_data_normalized = test_data_normalized.copy()

#                 # 将生成器输出展开为一维数组
#                 generated_values = generated_values.flatten()

#                 # 按缺失值掩码逐行填充缺失值
#                 missing_indices = missing_mask.index[missing_mask[test_column]].tolist()
#                 for idx, value in zip(missing_indices, generated_values):
#                     filled_data_normalized.loc[idx, test_column] = value

#                 # 反标准化结果
#                 filled_data = pd.DataFrame(
#                     scaler.inverse_transform(filled_data_normalized),
#                     columns=feature_columns
#                 )

#                 # 分析误差
#                 mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(
#                     data_extend_test_ori.iop_os_values,
#                     filled_data[test_column],
#                     data_extend_val.iop_os_values
#                 )
#                 results_Process_Inter_GAN.iop_os.predicted_list.append(predicted_value)
#                 results_Process_Inter_GAN.iop_os.true_list.append(true_value)


#             except Exception as e:
#                 print(f"处理文件 {filename} 时出错: {e}")
#                 traceback.print_exc()


#     # 汇总结果并保存为 Excel

#     results_Process_Inter_GAN.iop_os.predicted_val = pd.concat(
#             results_Process_Inter_GAN.iop_os.predicted_list, ignore_index=True)
#     results_Process_Inter_GAN.iop_os.true_val = pd.concat(
#             results_Process_Inter_GAN.iop_os.true_list, ignore_index=True)

#     mse, mae, accuracy = Func_Analyse_Evaluate(
#             results_Process_Inter_GAN.iop_os.predicted_val,
#             results_Process_Inter_GAN.iop_os.true_val
#         )

#     # 保存结果
#     df = pd.DataFrame({
#             'predicted_val': results_Process_Inter_GAN.iop_os.predicted_val,
#             'true_val': results_Process_Inter_GAN.iop_os.true_val,
#             'mse': mse,
#             'mae': mae,
#             'Acc': accuracy
#         })
#     df.to_excel(output_path + f'/Analyse_GAN_process.xlsx', index=False, header=True)
#     print(mse,mae,accuracy)


# def Func_train_gan_multimodal(generator, discriminator, gan, normalized_data, missing_mask,input_dim):
#     """
#     训练多模态 GAN 模型。

#     输入：
#     - generator: 生成器模型。
#     - discriminator: 判别器模型。
#     - gan: GAN 整体模型。
#     - data (pd.DataFrame): 标准化后的训练数据。
#     - missing_mask (pd.DataFrame): 缺失值掩码。
#     - epochs (int): 训练轮数。
#     - batch_size (int): 批量大小。

#     输出：
#     - generator: 训练后的生成器。
#     """
#     # 提取非缺失行作为真实数据
#     # non_missing_data = normalized_data[~missing_mask.any(axis=1)].values
#     epochs = 1000
#     batch_size = 16


#     for epoch in range(epochs):
#         # 随机选择一个批次的数据
#         batch_indices = np.random.choice(normalized_data.shape[0], batch_size, replace=True)
#         real_data = normalized_data.iloc[batch_indices].values
#         real_data = normalized_data.iloc[batch_indices].values
#         real_mask = missing_mask.iloc[batch_indices].values

#         # 判别器训练
#         noise = np.random.normal(0, 1, size=(batch_size, input_dim))
#         generated_data = generator.predict(noise)

#         # 将生成的数据合并到真实数据中，替换缺失部分
#         combined_data = real_data.copy()
#         combined_data[real_mask] = generated_data[real_mask]

#         X_discriminator = np.concatenate([real_data, combined_data], axis=0)
#         y_discriminator = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=0)

#         # 调试信息
#         print(f"X_discriminator shape: {X_discriminator.shape}")
#         print(f"y_discriminator shape: {y_discriminator.shape}")

#         discriminator.trainable = True
#         # 判别器训练

#         d_loss, d_acc = discriminator.train_on_batch(X_discriminator, y_discriminator)
#         print(f"判别器训练完成: D_loss={d_loss:.4f}, D_acc={d_acc:.4f}")


#         # 生成器训练
#         noise = np.random.normal(0, 1, size=(batch_size, input_dim))
#         y_generator = np.ones((batch_size, 1))  # 欺骗判别器的标签
#         discriminator.trainable = False
#         g_loss = gan.train_on_batch(noise, y_generator)

#         # 打印训练信息
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}/{epochs} | D_loss: {d_loss:.4f}, D_acc: {d_acc:.4f}, G_loss: {g_loss:.4f}")

#     return generator

# 生成器（Generator）
def build_generator(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(input_shape[0], activation='sigmoid')  # 使用sigmoid保证输出在0到1之间
    ])
    return model

# 判别器（Discriminator）
def build_discriminator(input_shape):
    """
    判别器网络：
    - 输入：填充后的数据 + Hint Mask
    - 输出：预测数据的真实性
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_shape[0] * 2,)),  # 修改为 2倍维度
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 预测是否为真实数据
    ])
    return model

# GAIN网络（生成器和判别器结合）
class GAIN_Model(tf.keras.Model):
    """
    GAIN模型，包含：
    - 生成器：填补缺失数据
    - 判别器：识别数据的真实性
    """
    def __init__(self, input_shape, name=None, trainable=True):
        super(GAIN_Model, self).__init__(name=name, trainable=trainable)
        self.input_shape_ = input_shape
        self.generator = build_generator(input_shape)
        self.discriminator = build_discriminator(input_shape)

    def call(self, inputs, training=False):
        data, mask, hint = inputs  # 新增 hint 作为输入
        generated_data = self.generator(data)  # 生成填补数据

        # 结合观测值和填充值
        filled_data = generated_data * (1 - mask) + data * mask

        # 判别器判断填补数据的真实性 (Hint机制引入)
        validity = self.discriminator(tf.concat([filled_data, hint], axis=1))  # 判别器接收 Hint 进行训练

        return filled_data, validity

    def get_config(self):
        """
        解决 Keras 模型保存和加载的问题
        """
        config = super(GAIN_Model, self).get_config()
        config.update({
            "input_shape": self.input_shape_,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        解决 Keras 模型加载的问题
        """
        # 提取必要的参数
        input_shape = config.pop("input_shape")
        # 保留keras模型基类需要的参数
        name = config.pop("name", None)
        trainable = config.pop("trainable", True)
        # 创建模型实例
        return cls(input_shape=input_shape, name=name, trainable=trainable)

# # 损失函数
# def gan_loss(y_true, y_pred):
#     y_true = tf.broadcast_to(y_true, tf.shape(y_pred))  # 确保形状匹配
#     return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)

def gan_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)

def mse_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

def total_loss(y_true, y_pred, generated_data, validity):
    return mse_loss(y_true, generated_data) + gan_loss(validity, y_pred)


def Func_Process_Multi_Inter_GAIN(extend_GAIN_train_path,extend_GAIN_val_path):
    """
    功能：
    使用 GAIN 网络填补缺失的数据并进行处理。

    输入：
    - extend_path (str): 包含原始数据文件的文件夹路径。
    - extend_val_path (str): 包含测试数据文件的文件夹路径。

    输出：
    - numpy.ndarray: 填充后的数据。
    """
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:/BaiduSyncdisk/QZZ/data_generation/output/GAIN_md/' + time_str
    model_path =  r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code\model'
    os.makedirs(output_path, exist_ok=True)
    features = ['gender_values', 'age_values', 'iop_values', 'cdr_values','md_values','rnfl_values']  #
    results_Process_Inter_GAIN = result()

    # 如果已有预训练模型，加载模型和标准化器
    path1 = model_path + '/gain_inter_md_model.h5'
    if os.path.exists(path1):
        with custom_object_scope({'GAIN_Model': GAIN_Model}):
            model = load_model(path1)
        scaler = joblib.load(model_path + '/gain_inter_md_scaler.pkl')
    else:
        # 如果没有模型，训练新的GAIN网络
        data_Process_Inter_GAIN_all = []

    # 遍历文件夹中的所有Excel文件
        for filename in os.listdir(extend_GAIN_train_path):
            if filename.startswith('patient_') and filename.endswith('.xlsx'):
                file_path = os.path.join(extend_GAIN_train_path, filename)
                try:
                    # 读取Excel文件
                    data_extend = Func_Dataloader_single(file_path)

                    # 填补数据缺失部分
                    for eye in ['od', 'os']:
                        test_data = pd.DataFrame({
                            'ID' :data_extend.ID,
                            'gender_values': data_extend.gender_values,
                            'age_values': data_extend.age_values,
                            'iop_values': getattr(data_extend, f'iop_{eye}_values'),
                            'cdr_values': getattr(data_extend, f'cdr_{eye}_values'),
                            'md_values': getattr(data_extend, f'md_{eye}_values'),
                            'rnfl_values': getattr(data_extend, f'rnfl_{eye}_values'),
                            # 'period_values': data_extend.period_values
                        })

                
                        data_Process_Inter_GAIN_all.append(pd.DataFrame(test_data))
                        

                except Exception as e:
                   print(f"处理文件 {filename} 时出错: {e}")
                   
                

                
        # 在这里执行标准化
        train_data = pd.concat(data_Process_Inter_GAIN_all, ignore_index=True)

                        # 标准化
        train_data_filled = train_data[features].copy()             
                        
        # 1. 创建缺失值的掩码矩阵（1表示观察到的值，0表示缺失值）
        mask = ~train_data_filled[features].isna()
        mask = mask.astype(np.float32)                
                        
        # 2. 将缺失值替换为0（这只是为了数值计算，GAIN会忽略这些位置）
        train_data_zeros = train_data_filled[features].fillna(0)                
                        
        scaler = StandardScaler()
        train_data_zeros = scaler.fit_transform(train_data_zeros)
        
        # 保存标准化器
        joblib.dump(scaler, output_path + 'gain_inter_md_scaler.pkl')

        # 创建 GAIN 模型
        # model = GAIN_Model(input_shape=(n_features,))

        model = GAIN_Model(input_shape=(train_data_zeros.shape[1],))  # 获取输入特征数量
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        #np版本
        X_train = train_data_zeros
        M_train = mask.to_numpy().astype(np.float32)



        # 训练参数
        epochs = 100
        batch_size = 64
        for epoch in range(epochs):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)

            # 小批量训练
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_data = X_train[batch_indices]
                batch_mask = M_train[batch_indices]
                
                        # 生成 Hint Mask（90% 的观测值作为 Hint）
                hint_mask = np.random.binomial(1, 0.9, batch_mask.shape)

                with tf.GradientTape(persistent=True) as tape:
                    generated_data = model.generator(batch_data, training=True)
                    filled_data = generated_data * (1 - batch_mask) + batch_data * batch_mask
                    validity = model.discriminator(tf.concat([filled_data, hint_mask], axis=1), training=True)

                    # 计算损失
                    loss = total_loss(batch_data, validity, generated_data, validity)

                # 计算梯度并更新参数
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 每 10 轮输出一次损失
            if epoch % 1 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.numpy():.4f}")

        # 保存模型和标准化器
        model.save(output_path +"gain_inter_md_model.h5")
        # joblib.dump(scaler, "scaler.pkl")
        print("模型训练完成，已保存！")

        




    for filename in os.listdir(extend_GAIN_val_path):
        # 确保只处理patient_开头的Excel文件
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            # 完整文件路径
            # file_path2 = os.path.join(extend_test_path, filename)
            file_path3 = os.path.join(extend_GAIN_val_path, filename)
            try:
                # 读取Excel文件
                # data_extend_test = Func_Dataloader_single(file_path2)
                # data_extend_test_ori = copy.copy(data_extend_test)
                data_extend_val = Func_Dataloader_single(file_path3)
                
                data_Process_Inter_GAIN_single = pd.DataFrame({
                    'ID': data_extend_val.ID,  # 左眼
                    'gender_values':data_extend_val.gender_values,
                    'age_values': data_extend_val.age_values,
                    'iop_values': data_extend_val.iop_od_values,
                    'cdr_values': data_extend_val.iop_od_values,
                    'md_values': data_extend_val.md_od_values,  # 替换 666 为 np.nan
                    'rnfl_values':data_extend_val.rnfl_od_values
                })
                
                data_extend_test = Func_create_missing_data_multi(data_Process_Inter_GAIN_single,['iop_values','md_values'])
                #给指定的列增加空数据，空数据值为666

                data_Process_Inter_GAIN_single = pd.DataFrame({
                    'ID': data_extend_test.ID,  # 左眼
                    'gender_values':data_extend_test.gender_values,
                    'age_values': data_extend_test.age_values,
                    'iop_values': data_extend_test.iop_values.replace(666, np.nan),
                    'cdr_values': data_extend_test.cdr_values,
                    'md_values': data_extend_test.md_values.replace(666, np.nan),  # 替换 666 为 np.nan
                    'rnfl_values': data_extend_test.rnfl_values,
                })
                GAIN_ori_data = data_Process_Inter_GAIN_single.copy()
                GAIN_test_data1 = data_Process_Inter_GAIN_single.copy()
                ###
                #标准化，需要填充
                
                        # train_data_zeros = train_data_filled[features].fillna(0)  
                
                test_mask = ~GAIN_test_data1[features].isna()
                test_mask = test_mask.astype(np.float32)       
                
                GAIN_test_data = scaler.transform(GAIN_test_data1[features].fillna(0))
                # GAIN_test = scaler.inverse_transform(GAIN_test_data)
                
                
                X_test = GAIN_test_data
                
                M_test = test_mask.to_numpy().astype(np.float32)
                # 预测缺失值

                hint_mask_test = M_test * np.random.binomial(1, 0.9, M_test.shape).astype(np.float32)
                
                                # 1. 运行 GAIN 生成器
                GAIN_test_data_f, _ = model([X_test, M_test, hint_mask_test], training=False)

                # 2. 使用 mask 直接合并观测值与填充值
                GAIN_test_data_filled = M_test * X_test + (1 - M_test) * GAIN_test_data_f  # 只填充缺失值

                # 3. 反标准化
                GAIN_test_data_filled = scaler.inverse_transform(GAIN_test_data_filled)

                # 4. 转换回 DataFrame
                GAIN_test_data_filled = pd.DataFrame(GAIN_test_data_filled, columns=features)

                
                iop_mse, iop_mae, iop_predicted_value, iop_true_value = Func_Analyse_Evaluate_Series(data_extend_test.iop_values,
                                                                                     GAIN_test_data_filled['iop_values'],
                                                                                     data_extend_val.iop_od_values)
                results_Process_Inter_GAIN.iop_od.predicted_list.append(iop_predicted_value)
                results_Process_Inter_GAIN.iop_od.true_list.append(iop_true_value)


                md_mse, md_mae, md_predicted_value, md_true_value = Func_Analyse_Evaluate_Series(data_extend_test.md_values,
                                                                                     GAIN_test_data_filled['md_values'],
                                                                                     data_extend_val.md_od_values)
                results_Process_Inter_GAIN.md_od.predicted_list.append(md_predicted_value)
                results_Process_Inter_GAIN.md_od.true_list.append(md_true_value)


            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()
                continue  
        # 保存结果



    results_Process_Inter_GAIN.iop_od.predicted_val = pd.concat(results_Process_Inter_GAIN.iop_od.predicted_list,ignore_index=True)
    results_Process_Inter_GAIN.iop_od.true_val = pd.concat(results_Process_Inter_GAIN.iop_od.true_list, ignore_index=True)
    mse,mae,accuracy = Func_Analyse_Evaluate(results_Process_Inter_GAIN.iop_od.predicted_val, results_Process_Inter_GAIN.iop_od.true_val)

    # 将 data_extend 转换为 DataFrame
    df = pd.DataFrame({
        'predicted_val': results_Process_Inter_GAIN.iop_od.predicted_val,
        'true_val': results_Process_Inter_GAIN.iop_od.true_val,
        'mse':mse,
        'mae':mae,
        'Acc':accuracy

    })

    # 输出成excel文件
    df.to_excel(output_path + '/Analyse_data_extend_iop_GAIN_process' + '.xlsx', index=False, header=True)

    results_Process_Inter_GAIN.md_od.predicted_val = pd.concat(results_Process_Inter_GAIN.md_od.predicted_list,ignore_index=True)
    results_Process_Inter_GAIN.md_od.true_val = pd.concat(results_Process_Inter_GAIN.md_od.true_list, ignore_index=True)
    mse,mae,accuracy = Func_md_Analyse_Evaluate(results_Process_Inter_GAIN.md_od.predicted_val, results_Process_Inter_GAIN.md_od.true_val)

    # 将 data_extend 转换为 DataFrame
    df = pd.DataFrame({
        'predicted_val': results_Process_Inter_GAIN.md_od.predicted_val,
        'true_val': results_Process_Inter_GAIN.md_od.true_val,
        'mse':mse,
        'mae':mae,
        'Acc':accuracy

    })

    # 输出成excel文件
    df.to_excel(output_path + '/Analyse_data_extend_md_GAIN_process' + '.xlsx', index=False, header=True)

    return print('over')



def Func_build_lstm_model(input_shape,learning_rate=0.001):
    """
    功能：
    构建一个用于预测的LSTM模型。

    输入：
    - input_shape (tuple): 模型输入的形状 (时间步数, 特征数)。
    - learning_rate (float, 默认值=0.001): 模型优化器的学习率。

    输出：
    - model (Sequential): 编译后的LSTM模型。
    """
    model = Sequential()
    model.add(Masking(mask_value=np.nan, input_shape=input_shape))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(1))  # 输出一个值，表示预测的iop值

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# def Func_Process_Inter_LSTM_preprocess_all():

def Func_train_lstm_model(x, y, window_size):
    """
    功能：
    使用给定的时间序列数据训练一个LSTM模型。

    输入：
    - x (numpy.ndarray): 输入特征数据，形状为 (样本数, 时间步数, 特征数)。
    - y (numpy.ndarray): 输出目标数据，形状为 (样本数, )。
    - window_size (int): 时间窗口的大小。

    输出：
    - model (Sequential): 经过训练的LSTM模型。
    """
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练
    model.fit(x, y, epochs=50, batch_size=32, verbose=2)
    return model



# def Func_Process_Inter_tGAN_Multimodal(extend_train_path,extend_test_path, extend_val_path, tgan_model_path=None):
#         """
#         使用多模态 GAN 模型填补缺失数据。

#         输入：
#         - extend_test_path (str): 测试数据路径。
#         - extend_val_path (str): 验证数据路径。
#         - gan_model_path (str): GAN 模型路径（可选，如果提供则加载预训练模型）。

#         输出：
#         - Excel 文件，包含 GANs 的预测结果和误差分析。
#         """
#         time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
#         output_path = f'E:/BaiduSyncdisk/QZZ/data_generation/output/tGAN_Multimodal_{time_str}'
#         os.makedirs(output_path, exist_ok=True)

#         # 特征列
#         feature_columns = ['gender_values', 'age_values', 'iop_od_values', 'iop_os_values', 'cdr_od_values',
#                            'cdr_os_values', 'md_od_values', 'md_os_values', 'rnfl_od_values', 'rnfl_os_values',
#                            'period_values']

        
        
#         # 滑动窗口参数
#         window_size = 4  # 每个时间序列的长度为7
#         stride = 1  # 每次滑动1步
#         FILL_VALUE = 666
#         # 创建一个空列表，用于存储用户数据
#         tGAN_all_data = []  # 存储未标准化的用户数据
#         missing_mask = []   # 存储缺失值掩码

#         # 如果有模型路径，则加载模型，否则构建一个新模型
#         tgan_model_path = None
#         if tgan_model_path:
#             generator = load_model(tgan_model_path)
#         else:
#             generator, discriminator, gan = Func_build_conditional_tgan_model(input_dim=len(feature_columns),fill_val=FILL_VALUE)

#             # 读取每个用户的数据（每个文件代表一个用户）
#             for filename in os.listdir(extend_train_path):
#                 if filename.startswith('patient_') and filename.endswith('.xlsx'):
#                     file_path = os.path.join(extend_train_path, filename)
#                     try:
#                         # 读取 Excel 文件
#                         data_extend = Func_Dataloader_single(file_path)


#                     except Exception as e:
#                         print(f"处理文件 {filename} 时出错: {e}")

#             # 将所有用户的处理数据合并成三维数组



#             # 训练 GANs


#             # 保存模型
#             generator.save(output_path + '/tGAN_inter_model.keras')

    # ##预测部分

def Func_build_conditional_tgan_model(input_dim, fill_val=666):
    """构建条件式TGAN模型"""

    
    # 生成器
    def build_generator():
        # 主输入：时间序列数据
        main_input = Input(shape=(None, input_dim))
        # 掩码输入：标记缺失值位置
        mask_input = Input(shape=(None, input_dim))
        
        # 合并输入
        merged = Concatenate()([main_input, mask_input])
        
        # 编码器
        x = LSTM(128, return_sequences=True)(merged)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        
        # 解码器
        x = TimeDistributed(Dense(32, activation='relu'))(x)
        x = TimeDistributed(Dense(input_dim, activation='tanh'))(x)
        
        return Model([main_input, mask_input], x, name='generator')
    
    # 判别器
    def build_discriminator():
        # 真实数据输入
        real_input = Input(shape=(None, input_dim))
        # 掩码输入
        mask_input = Input(shape=(None, input_dim))
        
        # 合并输入
        merged = Concatenate()([real_input, mask_input])
        
        # 特征提取
        x = LSTM(128, return_sequences=True)(merged)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = Dropout(0.3)(x)
        
        # 判别输出
        output = Dense(1, activation='sigmoid')(x)
        
        return Model([real_input, mask_input], output, name='discriminator')
    
    # 构建完整GAN
    def build_gan(generator, discriminator):
        discriminator.trainable = False
        
        # GAN输入
        gan_input = Input(shape=(None, input_dim))
        mask_input = Input(shape=(None, input_dim))
        
        # 生成器输出
        generated = generator([gan_input, mask_input])
        
        # 判别器判断生成数据
        validity = discriminator([generated, mask_input])
        
        return Model([gan_input, mask_input], validity, name='gan')
    
    # 实例化模型
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    return generator, discriminator, gan

def Func_train_tgan_conditional(generator, discriminator, gan, train_data, real_mask, 
                              input_dim, batch_size=16, epochs=100, missing_rate=0.2, 
                              fill_value=666):
    """训练条件式TGAN模型"""
    
    # 编译模型
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    # 训练参数
    batch_count = len(train_data) // batch_size
    
    # 训练循环
    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        
        for batch in range(batch_count):
            # 获取批次数据
            idx = np.random.randint(0, len(train_data), batch_size)
            real_batch = train_data[idx]
            mask_batch = real_mask[idx]
            
            # 创建噪声数据
            noise = np.random.normal(0, 1, (batch_size, real_batch.shape[1], input_dim))
            
            # 生成假数据
            gen_data = generator.predict([noise, mask_batch])
            
            # 训练判别器
            d_loss_real = discriminator.train_on_batch(
                [real_batch, mask_batch],
                np.ones((batch_size, 1))
            )
            d_loss_fake = discriminator.train_on_batch(
                [gen_data, mask_batch],
                np.zeros((batch_size, 1))
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # 训练生成器
            g_loss = gan.train_on_batch(
                [noise, mask_batch],
                np.ones((batch_size, 1))
            )
            
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"D Loss: {np.mean(d_losses):.4f}")
            print(f"G Loss: {np.mean(g_losses):.4f}")
    
    return generator, discriminator, gan

def Func_Process_Inter_tGAN_Multimodal(extend_train_path, extend_test_path, extend_val_path, tgan_model_path=None):
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = f'E:/BaiduSyncdisk/QZZ/data_generation/output/tGAN_Multimodal_{time_str}'
    os.makedirs(output_path, exist_ok=True)

    # 简化特征列
    feature_columns = ['gender_values', 'age_values', 'iop_values', 'period_values']
    FILL_VALUE = 666
    window_size = 4
    stride = 1
    
    # 存储处理后的数据
    tGAN_all_data = []
    tgan_model_path = r'E:\BaiduSyncdisk\QZZ\data_generation\output\tGAN_Multimodal_20250206_1740'
    if tgan_model_path:
        generator = load_model(os.path.join(tgan_model_path, 'tGAN_inter_model.keras'))
        norm_params = np.load(os.path.join(tgan_model_path, 'normalization_params.npz'))
    else:
        generator, discriminator, gan = Func_build_conditional_tgan_model(input_dim=len(feature_columns),
            fill_val=FILL_VALUE
        )
        
        
        # 处理训练数据
        for filename in os.listdir(extend_train_path):
            if filename.startswith('patient_') and filename.endswith('.xlsx'):
                file_path = os.path.join(extend_train_path, filename)
                try:
                    # 加载数据
                    data_extend = Func_Dataloader_single(file_path)
                    
                    # 合并左右眼数据
                    combined_data = []
                    for eye in ['od', 'os']:
                        df = pd.DataFrame({
                            'gender_values': data_extend.gender_values,
                            'age_values': data_extend.age_values,
                            'iop_values': getattr(data_extend, f'iop_{eye}_values'),
                            'period_values': data_extend.period_values
                        })
                        # 前向填充性别和时间间隔
                        df['gender_values'] = df['gender_values'].ffill()
                        df['period_values'] = df['period_values'].ffill()
                        combined_data.append(df)
                    
                    # 合并数据
                    patient_data = pd.concat(combined_data, ignore_index=True)
                    
                    # 创建滑动窗口序列
                    for i in range(0, len(patient_data) - window_size + 1, stride):
                        window = patient_data.iloc[i:i+window_size][feature_columns].values
                        if not np.all(np.isnan(window)):  # 排除全是NaN的窗口
                            tGAN_all_data.append(window)
                            
                except Exception as e:
                    print(f"处理训练文件 {filename} 时出错: {e}")
                    traceback.print_exc()
        
        # 转换为numpy数组
        tGAN_all_data = np.array(tGAN_all_data)
        
        # 创建缺失值掩码
        missing_mask = (tGAN_all_data == FILL_VALUE) | np.isnan(tGAN_all_data)
        real_mask = ~missing_mask
        
        # 标准化数据
        means = np.nanmean(np.where(real_mask, tGAN_all_data, np.nan), axis=(0,1))
        stds = np.nanstd(np.where(real_mask, tGAN_all_data, np.nan), axis=(0,1))
        normalized_data = (tGAN_all_data - means) / stds
        normalized_data = np.nan_to_num(normalized_data, nan=0)
        
        # 训练模型
        generator, discriminator, gan = Func_train_tgan_conditional(
            generator, discriminator, gan,
            normalized_data, real_mask=real_mask.astype(np.float32),
            input_dim=len(feature_columns),
            batch_size=16,
            epochs=100,
            missing_rate=0.2,
            fill_value=FILL_VALUE
        )
        
        # 保存模型和标准化参数
        generator.save(os.path.join(output_path, 'tGAN_inter_model.keras'))
        np.savez(os.path.join(output_path, 'normalization_params.npz'), 
                 means=means, stds=stds)
    
    # 处理测试数据
    means, stds = norm_params['means'], norm_params['stds']
    for filename in os.listdir(extend_test_path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path = os.path.join(extend_test_path, filename)
            try:
                data_test = Func_Dataloader_single(file_path)
                
                for eye in ['od', 'os']:
                    test_data = pd.DataFrame({
                        'gender_values': data_test.gender_values,
                        'age_values': data_test.age_values,
                        'iop_values': getattr(data_test, f'iop_{eye}_values'),
                        'period_values': data_test.period_values
                    })
                    
                    # 前向填充固定值
                    test_data['gender_values'] = test_data['gender_values'].ffill()
                    test_data['period_values'] = test_data['period_values'].ffill()
                    
                    # 获取原始IOP序列长度
                    original_length = len(getattr(data_test, f'iop_{eye}_values'))
                    
                    # 创建滑动窗口序列
                    test_sequences = []
                    test_masks = []
                    
                    for i in range(0, len(test_data) - window_size + 1, stride):
                        window = test_data.iloc[i:i+window_size].values
                        mask = ~np.isnan(window)
                        test_sequences.append(window)
                        test_masks.append(mask)
                    
                    test_sequences = np.array(test_sequences)
                    test_masks = np.array(test_masks).astype(np.float32)
                    
                    # 标准化和生成
                    test_normalized = (test_sequences - means) / stds
                    test_normalized = np.nan_to_num(test_normalized, nan=0)
                    generated = generator.predict([test_normalized, test_masks])
                    
                    # 反标准化
                    filled_values = generated * stds[2] + means[2]  # 只取iop值
                    
                    # 创建结果数组
                    filled_series = np.zeros(original_length)
                    filled_series[:] = np.nan
                    
                    # 填充缺失值
                    iop_col = f'iop_{eye}_values'
                    original_values = getattr(data_test, iop_col)
                    mask = np.isnan(original_values)
                    
                    # 对每个窗口的预测值进行处理
                    for i in range(len(test_sequences)):
                        window_start = i
                        window_end = i + window_size
                        # 只在缺失位置填充预测值
                        for j in range(window_size):
                            if mask[window_start + j]:
                                filled_series[window_start + j] = filled_values[i][j]
                    
                    # 更新数据
                    # 只在原始值为nan的位置使用填充值
                    final_values = np.where(mask, 
                                        filled_series, 
                                        original_values)
                    setattr(data_test, iop_col, final_values)
                    
                # 保存填充后的数据
                output_file = os.path.join(output_path, f'filled_{filename}')
                
                
                data_test.to_excel(output_file, index=False)
                print(f"已保存填充后的文件: {output_file}")
                
            except Exception as e:
                print(f"处理测试文件 {filename} 时出错: {e}")
                traceback.print_exc()

def Func_Inter_iopcdr_LSTM(extend_path,extend_output_path):
    """
    功能：
    使用LSTM模型处理数据并预测缺失值。

    输入：
    - extend_path: 训练数据路径
    - output_path: 输出数据路径
    """
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output\LSTM_iopcdr' + '/' + time_str
    model_path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code\model'
    results_Process_Inter_LSTM = result()
    data_Process_Inter_LSTM_all = []


    iop_model = load_model(model_path + '\lstm_inter_model.keras')
    iop_scaler = joblib.load(model_path + '\lstm_inter_cdr_scaler.pkl')
    
    cdr_model = load_model(model_path + '\lstm_inter_cdr_model.keras')
    cdr_scaler = joblib.load(model_path + '\lstm_inter_cdr_scaler.pkl')

    for filename in os.listdir(extend_path):
        # 确保只处理patient_开头的Excel文件
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path = os.path.join(extend_path, filename)
            try:
                # 读取原始Excel文件
                df = pd.read_excel(file_path)
                
                # 获取需要填充的列
                iop_cols = ['iop_od_values', 'iop_os_values']
                cdr_cols = ['cdr_od_values', 'cdr_os_values']
                
                # 复制原始数据用于填充
                df_filled = df.copy()
                
                # 对IOP数据进行填充
                for col in iop_cols:
                    # 找出需要填充的位置(值为666)
                    mask = pd.isna(df[col])
                    if mask.any():
                        # 准备数据
                        data = pd.DataFrame({
                            'age_values': df['age_values'],
                            'iop_values': df[col]
                        })
                        
                        # 标准化数据
                        data['age_values'] = cdr_scaler.transform(data['age_values'].values.reshape(-1, 1))
                        data['iop_values'] = iop_scaler.transform(data['iop_values'].values.reshape(-1, 1))
                        
                        # 预测填充
                        window_size = 3
                        filled_data = Func_LSTM_predict_missing_values(iop_model, data, window_size, iop_scaler, 'iop_values')
                        
                        # 反标准化
                        filled_values = Func_reverse_standardization(iop_scaler, filled_data['iop_values'].values)
                        filled_values[filled_values < 8] = 8
                        
                        # 只在666的位置进行填充
                        df_filled.loc[mask, col] = filled_values[mask]
                
                # 对CDR数据进行填充
                for col in cdr_cols:
                    # 找出需要填充的位置(值为nan)
                    mask = pd.isna(df[col]) 
                    if mask.any():
                        # 准备数据
                        data = pd.DataFrame({
                            'age_values': df['age_values'],
                            'cdr_values': df[col]
                        })
                        
                        # 标准化数据
                        data['age_values'] = cdr_scaler.transform(data['age_values'].values.reshape(-1, 1))
                        data['cdr_values'] = cdr_scaler.transform(data['cdr_values'].values.reshape(-1, 1))
                        
                        # 预测填充
                        window_size = 3
                        filled_data = Func_LSTM_predict_missing_values(cdr_model, data, window_size, cdr_scaler, 'cdr_values')
                        
                        # 反标准化
                        filled_values = Func_reverse_standardization(cdr_scaler, filled_data['cdr_values'].values)
                        filled_values[filled_values < 0.2] = 0.2
                        
                        # 只在nan的位置进行填充
                        df_filled.loc[mask, col] = filled_values[mask]
                
                # 保存填充后的文件
                os.makedirs(output_path, exist_ok=True)
                output_file = os.path.join(output_path, f'{filename}')
                df_filled.to_excel(output_file, index=False)
                print(f"已保存填充后的文件: {output_file}")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()


def Func_Process_Inter_LSTM(extend_path,extend_val_path):
    """
    功能：
    将标准化后的数据反向转换为原始值。

    输入：
    - scaler (StandardScaler): 用于标准化数据的缩放器。
    - standardized_values (numpy.ndarray): 标准化后的值。

    输出：
    - numpy.ndarray: 反标准化后的原始值。
    """
    # 创建输出图表的文件夹（如果不存在）

    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output\LSTM_md' + '/' + time_str
    model_path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code\model'
    results_Process_Inter_LSTM = result()
    data_Process_Inter_LSTM_all = []

    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(model_path+ '\lstm_inter_md_model.keras'):
        # os.makedirs(model_path)
        model = load_model(model_path + '\lstm_inter_md_model.keras')
        scaler = joblib.load(model_path + '\lstm_inter_md_scaler.pkl')
    else:
        # 遍历文件夹中的所有Excel文件
        for filename in os.listdir(extend_path):
            # 确保只处理patient_开头的Excel文件
            if filename.startswith('patient_') and filename.endswith('.xlsx'):
                # 完整文件路径
                file_path1 = os.path.join(extend_path, filename)
                # file_path2 = os.path.join(extend_empty_path, filename)
                try:
                    # 读取Excel文件

                    data_extend = Func_Dataloader_single(file_path1)


                    data_Process_Inter_LSTM_single = {}
                    data_Process_Inter_LSTM_single['ID'] = data_extend.ID * 10 + 1              ##左眼
                    data_Process_Inter_LSTM_single['age_values'] = data_extend.age_values
                    data_Process_Inter_LSTM_single['md_values'] = data_extend.md_od_values
                    data_Process_Inter_LSTM_all.append(pd.DataFrame(data_Process_Inter_LSTM_single))

                    data_Process_Inter_LSTM_single = {}
                    data_Process_Inter_LSTM_single['ID'] = data_extend.ID * 10 + 2              ##右眼
                    data_Process_Inter_LSTM_single['age_values'] = data_extend.age_values
                    data_Process_Inter_LSTM_single['md_values'] = data_extend.md_os_values
                    data_Process_Inter_LSTM_all.append(pd.DataFrame(data_Process_Inter_LSTM_single))



                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")


        # model = load_model(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code\lstm_inter_model.keras')
        LSTM_all_data = pd.concat(data_Process_Inter_LSTM_all, axis=0, ignore_index=True)
        #标准化
        scaler = StandardScaler()
        # 对非缺失值进行标准化
        LSTM_all_data['age_values'] = scaler.fit_transform(LSTM_all_data['age_values'].values.reshape(-1, 1))
        LSTM_all_data['md_values'] = scaler.fit_transform(LSTM_all_data['md_values'].values.reshape(-1, 1))


        # 切割
        # 切片数据
        window_size = 3
        X, y = Func_preprocess_and_slice_data(LSTM_all_data, window_size)

        # 确保输入数据为 LSTM 格式 (样本数, 时间步长, 特征数)
        X = X.reshape((X.shape[0], X.shape[1], 2))  # 2个特征 (age 和 iop_values)

        # 创建模型
        model = Func_build_lstm_model(input_shape=(window_size, 2))

        # 训练模型
        model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=2)
        #
        
        model.save(output_path +'\lstm_inter_md_model.keras')  # 默认保存模型和权重
        joblib.dump(scaler, output_path +'\lstm_inter_md_scaler.pkl')


    data_Process_Inter_LSTM_test = []

##读取测试数据，并且把666替换为nan
    for filename in os.listdir(extend_val_path):
        # 确保只处理patient_开头的Excel文件
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            # 完整文件路径
            # file_path2 = os.path.join(extend_test_path, filename)
            file_path3 = os.path.join(extend_val_path, filename)
            try:
                # 读取Excel文件
                # data_extend_test = Func_Dataloader_single(file_path2)
                # data_extend_test_ori = copy.copy(data_extend_test)
                data_extend_val = Func_Dataloader_single(file_path3)
                
                data_extend_test = Func_create_missing_data(copy.copy(data_extend_val))
                data_extend_test_ori = copy.copy(data_extend_test)

                data_Process_Inter_LSTM_single = pd.DataFrame({
                    'ID': data_extend_test.ID,  # 左眼
                    'age_values': data_extend_test.age_values,
                    'md_values': data_extend_test.md_od_values.replace(666, np.nan)  # 替换 666 为 np.nan
                })

                LSTM_test_data = data_Process_Inter_LSTM_single.copy()
                LSTM_test_data['age_values'] = scaler.transform(LSTM_test_data['age_values'].values.reshape(-1, 1))
                LSTM_test_data['md_values'] = scaler.transform(LSTM_test_data['md_values'].values.reshape(-1, 1))
                # 预测缺失值
                window_size = 3
                filled_data = Func_LSTM_predict_missing_values(model, LSTM_test_data, window_size, scaler,value_name='md_values')
                # 3. 填补缺失值后，对结果进行反标准化
                # 将填补后的iop_values列进行反标准化
                filled_data['md_values'] = Func_reverse_standardization(scaler, filled_data['md_values'].values)

                # 4. 如果需要，原始的age_values也可以进行反标准化（通常可以直接使用原始的age_values，因为它没有缺失）
                filled_data['age_values'] = data_Process_Inter_LSTM_single['age_values']

                mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_test_ori.md_od_values,
                                                                                     filled_data['md_values'],
                                                                                     data_extend_val.md_od_values)
                results_Process_Inter_LSTM.md_od.predicted_list.append(predicted_value)
                results_Process_Inter_LSTM.md_od.true_list.append(true_value)



                data_Process_Inter_LSTM_single = pd.DataFrame({
                    'ID': data_extend_test.ID * 10,  # 右眼
                    'age_values': data_extend_test.age_values,
                    'md_values': data_extend_test.md_os_values.replace(666, np.nan)  # 替换 666 为 np.nan
                })



                LSTM_test_data = data_Process_Inter_LSTM_single.copy()
                LSTM_test_data['age_values'] = scaler.transform(LSTM_test_data['age_values'].values.reshape(-1, 1))
                LSTM_test_data['md_values'] = scaler.transform(LSTM_test_data['md_values'].values.reshape(-1, 1))
                # 预测缺失值
                filled_data = Func_LSTM_predict_missing_values(model, LSTM_test_data, window_size, scaler,value_name='md_values')
                # 3. 填补缺失值后，对结果进行反标准化
                # 将填补后的iop_values列进行反标准化
                filled_data['md_values'] = Func_reverse_standardization(scaler, filled_data['md_values'].values)

                # 4. 如果需要，原始的age_values也可以进行反标准化（通常可以直接使用原始的age_values，因为它没有缺失）
                filled_data['age_values'] = data_Process_Inter_LSTM_single['age_values']

                mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_test_ori.md_os_values,
                                                                                     filled_data['md_values'],
                                                                                     data_extend_val.md_os_values)
                results_Process_Inter_LSTM.md_os.predicted_list.append(predicted_value)
                results_Process_Inter_LSTM.md_os.true_list.append(true_value)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()



    results_Process_Inter_LSTM.md_od.predicted_val = pd.concat(results_Process_Inter_LSTM.md_od.predicted_list,ignore_index=True)
    results_Process_Inter_LSTM.md_od.true_val = pd.concat(results_Process_Inter_LSTM.md_od.true_list, ignore_index=True)
    mse,mae,accuracy = Func_md_Analyse_Evaluate(results_Process_Inter_LSTM.md_od.predicted_val, results_Process_Inter_LSTM.md_od.true_val)

    # 将 data_extend 转换为 DataFrame
    df = pd.DataFrame({
        'predicted_val': results_Process_Inter_LSTM.md_od.predicted_val,
        'true_val': results_Process_Inter_LSTM.md_od.true_val,
        'mse':mse,
        'mae':mae,
        'Acc':accuracy

    })

    # 输出成excel文件
    df.to_excel(output_path + '/Analyse_data_extend_md_LSTM_process' + '.xlsx', index=False, header=True)


def Func_Build_Multi_LSTM_Model(input_shape, output_size):
    """构建多输入多输出的 LSTM 模型"""
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_size))  # 输出层大小根据目标数量调整
    model.compile(optimizer='adam', loss='mse')
    return model



def Func_Process_Multi_Inter_LSTM(extend_path, extend_test_path, extend_val_path):
    """
    功能：
    使用多输入多输出 LSTM 模型处理数据并预测缺失值。

    输入：
    - extend_path: 训练数据路径
    - extend_test_path: 测试数据路径
    - extend_val_path: 验证数据路径

    输出：
    - 保存预测结果和模型到指定路径
    """
    # 创建输出文件夹
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
    output_path = r'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str
    os.makedirs(output_path, exist_ok=True)

    # 初始化结果存储
    results_Process_Inter_LSTM = result()
    data_Process_Inter_LSTM_all = []

    # 遍历训练数据文件
    for filename in os.listdir(extend_path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path = os.path.join(extend_path, filename)
            try:
                data_extend = Func_Dataloader_single(file_path)

                # ['age_values', 'diagnosis_values', 'iop_od_values', 'iop_os_values',
                #  'cdr_od_values', 'cdr_os_values', 'md_od_values', 'md_os_values',
                #  'rnfl_od_values', 'rnfl_os_values']

                # 左眼数据
                data_single = {
                    'ID': data_extend.ID * 10 + 1,
                    'gender_values':data_extend.gender_values,
                    'age_values': data_extend.age_values,
                    'iop_values': data_extend.iop_od_values,
                    'cdr_values': data_extend.cdr_od_values,  # 假设杯盘比字段
                    'md_values': data_extend.md_od_values,      # 假设视野字段
                    'rnfl_values': data_extend.rnfl_od_values,     # 假设时间间隔字段
                    'period_values':data_extend.period_values
                }
                data_Process_Inter_LSTM_all.append(pd.DataFrame(data_single))

                # 右眼数据
                data_single = {
                    'ID': data_extend.ID * 10 + 2,
                    'gender_values': data_extend.gender_values,
                    'age_values': data_extend.age_values,
                    'iop_values': data_extend.iop_os_values,
                    'cdr_values': data_extend.cdr_os_values,  # 假设杯盘比字段
                    'md_values': data_extend.md_os_values,      # 假设视野字段
                    'rnfl_values': data_extend.rnfl_os_values,     # 假设时间间隔字段
                    'period_values':data_extend.period_values
                }
                data_Process_Inter_LSTM_all.append(pd.DataFrame(data_single))

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()

    # 合并所有数据
    LSTM_all_data = pd.concat(data_Process_Inter_LSTM_all, axis=0, ignore_index=True)

    # 标准化
    scaler = StandardScaler()
    features = ['gender_values','age_values', 'iop_values', 'cdr_values', 'md_values','rnfl_values','period_values']
    targets= ['iop_values', 'cdr_values', 'md_values','rnfl_values']
    for feature in features:
        LSTM_all_data[feature] = scaler.fit_transform(LSTM_all_data[feature].values.reshape(-1, 1))
    for target in targets:
        LSTM_all_data[target] = scaler.fit_transform(LSTM_all_data[target].values.reshape(-1, 1))
    # 数据切分
    window_size = 3
    X, y = Func_Multi_preprocess_and_slice_data(LSTM_all_data, window_size)
    X = X.reshape((X.shape[0], X.shape[1], len(features) + len(targets)))

    # 构建模型
    model = Func_Build_Multi_LSTM_Model(input_shape=(window_size, len(features) + len(targets)), output_size=len(targets))

    # 训练模型
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=2)
    model.save(output_path + 'lstm_multi_inter_model.keras')

    def Func_LSTM_Multi_predict_missing_values(model, data, window_size, scaler):
        """
        功能：
        使用多输入多输出 LSTM 模型预测缺失值。

        输入：
        - model: 训练好的 LSTM 模型。
        - data (pd.DataFrame): 标准化后的测试数据。
        - window_size (int): 时间窗口大小。
        - scaler: 标准化器。

        输出：
        - filled_data (pd.DataFrame): 包含预测值的 DataFrame。
        """
        # 数据切分
        X, _ = Func_preprocess_and_slice_data(data, window_size)
        X = X.reshape((X.shape[0], X.shape[1], -1))

        # 预测
        predictions = model.predict(X)

        # 反标准化
        predictions = scaler.inverse_transform(predictions)

        # 创建结果 DataFrame
        filled_data = pd.DataFrame(predictions, columns=['iop_values', 'cdr_values', 'md_values', 'rnfl_values'])

        return filled_data

    # 测试数据处理
    for filename in os.listdir(extend_test_path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path_test = os.path.join(extend_test_path, filename)
            file_path_val = os.path.join(extend_val_path, filename)
            try:
                data_test = Func_Dataloader_single(file_path_test)
                data_test_ori = copy.copy(data_test)
                data_val = Func_Dataloader_single(file_path_val)

                # 左眼数据
                test_data = {
                    'ID': data_test.ID,
                    'gender_values': data_extend.gender_values,
                    'age_values': data_extend.age_values,
                    'iop_values': data_extend.iop_od_values,
                    'cdr_values': data_extend.cdr_od_values,  # 假设杯盘比字段
                    'md_values': data_extend.md_od_values,      # 假设视野字段
                    'rnfl_values': data_extend.rnfl_od_values,     # 假设时间间隔字段
                    'period_values':data_extend.period_values
                }
                test_df = pd.DataFrame(test_data)

                # 标准化
                for feature in features:
                    test_df[feature] = scaler.transform(test_df[feature].values.reshape(-1, 1))
                for target in targets:
                    test_df[target] = scaler.transform(test_df[target].values.reshape(-1, 1))

                # 预测缺失值
                filled_data = Func_LSTM_predict_missing_values(model, test_df, window_size, scaler)

                # 反标准化
                filled_data['iop_values'] = Func_reverse_standardization(scaler, filled_data['iop_values'].values)
                filled_data['cdr_values'] = Func_reverse_standardization(scaler, filled_data['cdr_values'].values)
                filled_data['md_values'] = Func_reverse_standardization(scaler, filled_data['md_values'].values)
                filled_data['rnfl_values'] = Func_reverse_standardization(scaler, filled_data['rnfl_values'].values)

                filled_data['gender_values'] = test_data['gender_values']
                filled_data['age_values'] = test_data['age_values']
                filled_data['period_values'] = test_data['period_values']
                # 评估
                mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(
                    data_test_ori.iop_od_values,
                    filled_data['iop_values'],
                    data_val.iop_od_values
                )
                results_Process_Inter_LSTM.iop_od['predicted_list'].append(predicted_value)
                results_Process_Inter_LSTM.iop_od['true_list'].append(true_value)

                # 右眼数据
                test_data = {
                    'ID': data_test.ID,
                    'gender_values': data_extend.gender_values,
                    'age_values': data_extend.age_values,
                    'iop_values': data_extend.iop_os_values,
                    'cdr_values': data_extend.cdr_os_values,  # 假设杯盘比字段
                    'md_values': data_extend.md_os_values,      # 假设视野字段
                    'rnfl_values': data_extend.rnfl_os_values,     # 假设时间间隔字段
                    'period_values':data_extend.period_values
                }
                test_df = pd.DataFrame(test_data)

                # 标准化
                for feature in features:
                    test_df[feature] = scaler.transform(test_df[feature].values.reshape(-1, 1))
                for target in targets:
                    test_df[target] = scaler.transform(test_df[target].values.reshape(-1, 1))

                # 预测缺失值
                filled_data = Func_LSTM_Multi_predict_missing_values(model, test_df, window_size, scaler)

                # 反标准化
                filled_data['iop_values'] = Func_reverse_standardization(scaler, filled_data['iop_values'].values)
                filled_data['cdr_values'] = Func_reverse_standardization(scaler, filled_data['cdr_values'].values)
                filled_data['md_values'] = Func_reverse_standardization(scaler, filled_data['md_values'].values)
                filled_data['rnfl_values'] = Func_reverse_standardization(scaler, filled_data['rnfl_values'].values)

                filled_data['gender_values'] = test_data['gender_values']
                filled_data['age_values'] = test_data['age_values']
                filled_data['period_values'] = test_data['period_values']

                # 评估
                mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(
                    data_test_ori.iop_os_values,
                    filled_data['iop_values'],
                    data_val.iop_os_values
                )
                results_Process_Inter_LSTM.iop_os['predicted_list'].append(predicted_value)
                results_Process_Inter_LSTM.iop_os['true_list'].append(true_value)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()

    # 保存结果
    df_od = pd.DataFrame({
        'predicted_val': results_Process_Inter_LSTM.iop_od['predicted_list'],
        'true_val': results_Process_Inter_LSTM.iop_od['true_list']
    })
    df_os = pd.DataFrame({
        'predicted_val': results_Process_Inter_LSTM.iop_os['predicted_list'],
        'true_val': results_Process_Inter_LSTM.iop_os['true_list']
    })

    mse_od, mae_od, accuracy_od = Func_Analyse_Evaluate(df_od['predicted_val'], df_od['true_val'])
    mse_os, mae_os, accuracy_os = Func_Analyse_Evaluate(df_os['predicted_val'], df_os['true_val'])

    df_od.to_excel(output_path + '/Analyse_data_od.xlsx', index=False, header=True)
    df_os.to_excel(output_path + '/Analyse_data_os.xlsx', index=False, header=True)




def Func_reverse_standardization(scaler, standardized_values):
    """
    反标准化数据，将标准化后的值还原为原始值。
    """
    return standardized_values * scaler.scale_[0] + scaler.mean_[0]

# 填补缺失值 (通过滑动窗口方式预测)
def Func_LSTM_predict_missing_values(model, test_data, window_size,scaler,value_name):
    """
    功能：
    使用LSTM模型预测数据中的缺失值。

    输入：
    - model (Sequential): 训练好的LSTM模型。
    - test_data (pd.DataFrame): 测试数据集，包含缺失值。
    - window_size (int): 用于预测的时间窗口大小。
    - scaler (StandardScaler): 用于数据标准化的缩放器。

    输出：
    - test_data (pd.DataFrame): 填补了缺失值的数据。
    """
    
    if value_name == 'cdr_values':
        fill_values = test_data['cdr_values'].values
    elif value_name == 'iop_values':
        fill_values = test_data['iop_values'].values
    elif value_name == 'md_values':
        fill_values = test_data['md_values'].values
    else:
        print('value_name 输入错误')
        return  

    age_values = test_data['age_values'].values

    # 将 666 替换为 NaN


    # 填补所有 NaN 的位置
    # 第一次预测
    for i in range(len(fill_values)):
        if np.isnan(fill_values[i]):  # 找到 NaN 的位置
            # 确保有足够的滑动窗口
            if i >= window_size:
                features = np.column_stack((age_values[i - window_size:i], fill_values[i - window_size:i]))
                features[:, 1] = np.nan_to_num(features[:, 1])  # 替换 NaN 为 0
                features = features.reshape((1, window_size, 2))  # 调整为 LSTM 输入格式

                # 使用模型预测
                predicted = model.predict(features, verbose=0)

                # 检查预测值是否为标量
                if predicted.size == 1:
                    fill_values[i] = predicted.item()  # 提取标量值并填补
                else:
                    print(f"预测值形状异常，位置: {i}, 预测值: {predicted}")

    # 第二次填补：向前/向后填补
    for i in range(len(fill_values)):
        if np.isnan(fill_values[i]):  # 找到剩余的 NaN
            # 尝试向后填补
            j = i + 1
            while j < len(fill_values) and np.isnan(fill_values[j]):
                j += 1
            if j < len(fill_values):  # 找到后面的非空值
                fill_values[i] = fill_values[j]
            else:
                # 如果无法向后填补，尝试向前填补
                j = i - 1
                while j >= 0 and np.isnan(fill_values[j]):
                    j -= 1
                if j >= 0:  # 找到前面的非空值
                    fill_values[i] = fill_values[j]

    # 返回填补后的数据
    test_data[value_name] = fill_values
    return test_data



def Func_preprocess_and_slice_data(data, window_size=3):
    """
    功能：
    将数据分组并切片为时间序列格式。

    输入：
    - data (pd.DataFrame): 包含时间序列数据的DataFrame。
    - window_size (int, 默认值=3): 时间窗口大小。

    输出：
    - X (numpy.ndarray): 输入特征数据。
    - y (numpy.ndarray): 输出目标数据。
    """
    X, y = [], []
    grouped = data.groupby('ID')  # 按照ID分组
    for _, group in grouped:
        # iop_values = group['iop_values'].values
        age_values = group['age_values'].values
        # gender_values = group['gender_values'].values
        cdr_values = group['md_values'].values
        # md_values = group['md_values'].values
        # rnfl_values = group['rnfl_values'].values
        # period_values = group['period_values'].values

        # 检查样本数量是否足够构建窗口
        if len(cdr_values) <= window_size:
            continue

        for i in range(len(cdr_values) - window_size):
            # 构建特征矩阵 (age_values 和 iop_values)
            features = np.column_stack((age_values[i:i + window_size], cdr_values[i:i + window_size]))
            target = cdr_values[i + window_size]  # 窗口后面的目标值

            # 跳过包含 NaN 的窗口或目标值
            if np.isnan(features).any() or np.isnan(target):
                continue

            X.append(features)
            y.append(target)

    return np.array(X), np.array(y)


def Func_Multi_preprocess_and_slice_data(data, window_size=3):
    """
    功能：
    将数据分组并切片为时间序列格式。

    输入：
    - data (pd.DataFrame): 包含时间序列数据的DataFrame。
    - window_size (int, 默认值=3): 时间窗口大小。

    输出：
    - X (numpy.ndarray): 输入特征数据，形状为 (样本数, 时间步长, 特征数)。
    - y (numpy.ndarray): 输出目标数据，形状为 (样本数, 目标数)。
    """
    X, y = [], []
    features = ['gender_values', 'age_values', 'iop_values', 'cdr_values', 'md_values', 'rnfl_values', 'period_values']
    targets = ['iop_values', 'cdr_values', 'md_values', 'rnfl_values']

    grouped = data.groupby('ID')  # 按照ID分组
    for _, group in grouped:
        # 获取特征和目标值
        feature_values = group[features].values
        target_values = group[targets].values

        # 检查样本数量是否足够构建窗口
        if len(feature_values) <= window_size:
            continue

        for i in range(len(feature_values) - window_size):
            # 构建特征矩阵 (时间步长, 特征数)
            features_window = feature_values[i:i + window_size]
            targets_window = target_values[i + window_size]  # 窗口后面的目标值

            # 跳过包含 NaN 的窗口或目标值
            if np.isnan(features_window).any() or np.isnan(targets_window).any():
                continue

            X.append(features_window)
            y.append(targets_window)

    return np.array(X), np.array(y)


def Func_Process_Inter_KNN_single(data_extend,data_extend_empty,results_Process_Inter_KNN,output_path):
    """
    功能：
    处理单个患者的数据，利用 KNN 算法对缺失值进行插值，计算插值结果的误差指标，并保存处理后的数据到指定路径。

    输入：
    - data_extend (Phandle): 包含完整患者数据的对象，用于作为真实值的参考。
    - data_extend_empty (Phandle): 包含缺失值的患者数据对象，用于进行插值处理。
    - results_Process_Inter_KNN (result): 存储插值结果和误差指标的对象。
    - output_path (str): 输出文件夹路径，用于保存处理后的结果。

    输出：
    无直接返回值。结果存储在 `results_Process_Inter_KNN` 对象中，处理后的数据以 Excel 文件形式保存到指定路径。

    详细步骤：
    """
    data_Process_Inter_KNN = copy.copy(data_extend)

    data_Process_Inter_KNN.md_od_values = Func_Algorithm_Inter_KNN(data_extend_empty.md_od_values)
    data_Process_Inter_KNN.md_os_values = Func_Algorithm_Inter_KNN(data_extend_empty.md_os_values)

    # data_Process_Inter_KNN.cdr_od_values = Func_Algorithm_Inter_KNN(data_extend_empty.cdr_od_values)
    # data_Process_Inter_KNN.cdr_os_values = Func_Algorithm_Inter_KNN(data_extend_empty.cdr_os_values)

    mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_empty.md_od_values,data_Process_Inter_KNN.md_od_values,data_extend.md_od_values)
    results_Process_Inter_KNN.md_od.predicted_list.append(predicted_value)
    results_Process_Inter_KNN.md_od.true_list.append(true_value)

    mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_empty.md_os_values,data_Process_Inter_KNN.md_os_values,data_extend.md_os_values)
    results_Process_Inter_KNN.md_os.predicted_list.append(predicted_value)
    results_Process_Inter_KNN.md_os.true_list.append(true_value)

    path = output_path + '/result_data_extend_KNN_md_process/'
    Func_output_excel(data_Process_Inter_KNN, path)
    
def Func_Process_Inter_KNN(extend_path, extend_empty_path):
    """
    path (str): 包含患者Excel文件的文件夹路径
    """
    # 创建输出图表的文件夹（如果不存在）
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str

    results_Process_Inter_KNN = result()


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
                # data_extend_empty = Func_create_missing_data(data_extend)
                data_extend_empty = Func_Dataloader_single(file_path2)
                
                Func_Process_Inter_KNN_single(data_extend, data_extend_empty, results_Process_Inter_KNN, output_path)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    results_Process_Inter_KNN.md_od.predicted_val = pd.concat(results_Process_Inter_KNN.md_od.predicted_list,ignore_index=True)
    results_Process_Inter_KNN.md_od.true_val = pd.concat(results_Process_Inter_KNN.md_od.true_list, ignore_index=True)
    mse,mae,accuracy = Func_md_Analyse_Evaluate(results_Process_Inter_KNN.md_od.predicted_val, results_Process_Inter_KNN.md_od.true_val)

    # 将 data_extend 转换为 DataFrame
    df = pd.DataFrame({
        'predicted_val': results_Process_Inter_KNN.md_od.predicted_val,
        'true_val': results_Process_Inter_KNN.md_od.true_val,
        'mse':mse,
        'mae':mae,
        'Acc':accuracy

    })

    # 输出成excel文件
    df.to_excel(output_path + '/Analyse_data_extend_KNN_md_process' + '.xlsx', index=False, header=True)



def Func_Dataloader2(path):
    """
    遍历文件夹中的患者数据文件,读取并处理每个患者的数据。
    主要功能包括:
    1. 读取Excel文件中的患者数据
    2. 提取并计算各项指标(眼压、杯盘比、视野等)
    3. 生成时间序列扩展数据
    4. 创建缺失数据用于后续分析

    参数:
    path (str): 包含患者Excel文件的文件夹路径

    处理流程:
    1. 创建输出目录
    2. 遍历文件夹中的Excel文件
    3. 对每个文件:
       - 读取数据到DataFrame
       - 初始化数据处理对象
       - 提取并处理各项指标数据
       - 计算差值指标
       - 扩展时间序列
       - 生成缺失数据
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
                phandle = Phandle()  # 初始化数据处理对象
                data_extend = Phandle()  # 用于存储扩展后的数据
                data_extend_empty = Phandle()  # 用于存储带有缺失值的数据

                phandle.ID = filename
                data_extend.ID = filename
                # 按照时间排序
                df_sorted = df.sort_values('treat_dates').reset_index(drop=True)

                phandle.ID = df_sorted.iloc[:, 0]

                phandle.gender_values = df_sorted.iloc[:, 1]  # 第十列 性别
                # phandle.gender_values = df_sorted.iloc[:, 1].apply(lambda x: 0 if x == '女' else 1)
                phandle.birth_dates = df_sorted.iloc[:, 2]
                phandle.age_values = df_sorted.iloc[:, 3]  # 第十列 年龄
                phandle.treat_dates = df_sorted.iloc[:, 5]

                phandle.diagnosis_values = df_sorted.iloc[:, 4]

                phandle.iop_od_values = df_sorted.iloc[:, 6]  # 第十列 眼压
                phandle.iop_os_values = df_sorted.iloc[:, 7]  # 第十一列

                #计算右眼眼压差值
                phandle.del_iop_od_values = [0] + [phandle.iop_od_values[i] - phandle.iop_od_values[i - 1] for i in
                                                   range(1, len(phandle.iop_od_values))]
                # 计算左眼眼压差值
                phandle.del_iop_os_values = [0] + [phandle.iop_os_values[i] - phandle.iop_os_values[i - 1] for i in
                                                   range(1, len(phandle.iop_os_values))]

                phandle.cdr_od_values = df_sorted.iloc[:, 8]  # 第十2列 杯盘比
                phandle.cdr_os_values = df_sorted.iloc[:, 9]  # 第十3列

                # 计算右眼杯盘比的差值，并在开头插入0以保持长度不变
                # 计算右眼杯盘比差值
                phandle.del_cdr_od_values = [0] + [phandle.cdr_od_values[i] - phandle.cdr_od_values[i - 1] for i in
                                                   range(1, len(phandle.cdr_od_values))]

                # 计算左眼杯盘比差值
                phandle.del_cdr_os_values = [0] + [phandle.cdr_os_values[i] - phandle.cdr_os_values[i - 1] for i in
                                                   range(1, len(phandle.cdr_os_values))]
                #
                phandle.md_od_values = df_sorted.iloc[:, 10]  # 第十2列  视野
                phandle.md_os_values = df_sorted.iloc[:, 11]  # 第十3列

                phandle.rnfl_od_values = df_sorted.iloc[:, 12]
                phandle.rnfl_os_values = df_sorted.iloc[:, 13]

                # data_extend = Func_time_series_extend_nearest(phandle, data_extend)
                data_extend_empty = Func_create_missing_data(phandle)
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

def Func_Process_Multi_Inter_random_forest(extend_path,extend_empty_path):

    """
    path (str): 包含患者Excel文件的文件夹路径
    """
    # 创建输出图表的文件夹（如果不存在）
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str

    results_Process_Inter_random_forest = result()


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
                # data_extend = Func_Dataloader_single(file_path1)
                # # data_extend_empty = Func_create_missing_data(data_extend)
                # data_extend_empty = Func_Dataloader_single(file_path2)
                
                data_extend = pd.read_excel(file_path1)
                data_extend_empty = pd.read_excel(file_path2)
                
                results_Process_Inter_random_forest = Func_Process_Inter_Random_Forest_single(data_extend, data_extend_empty, results_Process_Inter_random_forest, output_path)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    results_Process_Inter_random_forest.iop_od.predicted_val = pd.concat(results_Process_Inter_random_forest.iop_od.predicted_list,ignore_index=True)
    results_Process_Inter_random_forest.iop_od.true_val = pd.concat(results_Process_Inter_random_forest.iop_od.true_list, ignore_index=True)
    mse,mae,accuracy = Func_Analyse_Evaluate(results_Process_Inter_random_forest.iop_od.predicted_val, results_Process_Inter_random_forest.iop_od.true_val)

    # 将 data_extend 转换为 DataFrame
    df = pd.DataFrame({
        'predicted_val': results_Process_Inter_random_forest.iop_od.predicted_val,
        'true_val': results_Process_Inter_random_forest.iop_od.true_val,
        'mse':mse,
        'mae':mae,
        'Acc':accuracy

    })

    # 输出成excel文件
    df.to_excel(output_path + '/Analyse_data_extend_randomforest_iop_process' + '.xlsx', index=False, header=True)

def Func_Algorithm_random_forest(data_df, features, n_estimators=100, max_depth=30):
    """
    使用随机森林进行多变量缺失值填充
    
    参数:
    data_df: pd.DataFrame, 包含缺失值的数据框
    features: list, 需要填充的特征列表,默认为None表示填充所有列
    n_estimators: int, 随机森林中树的数量
    max_depth: int, 树的最大深度,None表示不限制深度
    
    返回:
    pd.DataFrame: 填充后的数据框
    """
    # 如果未指定特征,则处理所有数值型列
    if features is None:
        features = data_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 复制原始数据
    imputed_df = data_df.copy()
    imputed_df.replace(666, np.nan, inplace=True)
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(imputed_df[features]), 
                           columns=features,
                           index=imputed_df.index)
    
    # 对每个含缺失值的特征进行填充
    for feature in features:
        # 获取当前特征的缺失值索引
        missing_idx = imputed_df[feature].isnull()
        
        if missing_idx.sum() > 0:  # 如果存在缺失值
            # 构建训练集 - 使用已知值的样本
            known_idx = ~missing_idx
            train_data = scaled_df[known_idx]
            
            # 分离特征值和目标值
            X_train = train_data.drop(columns=[feature])
            y_train = train_data[feature]
            
            # 训练随机森林模型
            rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           random_state=42)
            rf_model.fit(X_train, y_train)
            
            # 对缺失值进行预测
            X_missing = scaled_df[missing_idx].drop(columns=[feature])
            predicted_values = rf_model.predict(X_missing)
            
            # 将预测值填充回原始数据框
            imputed_df.loc[missing_idx, feature] = scaler.inverse_transform(
                scaled_df.loc[missing_idx].assign(**{feature: predicted_values})
            )[:, features.index(feature)]
            
    return imputed_df

# 使用示例:
def Func_Process_Inter_Random_Forest_single(data_extend,data_extend_empty,results_Process_Inter_Random_Forest,output_path):
    """
    功能：
    处理单个患者的数据，利用 KNN 算法对缺失值进行插值，计算插值结果的误差指标，并保存处理后的数据到指定路径。

    输入：
    - data_extend (Phandle): 包含完整患者数据的对象，用于作为真实值的参考。
    - data_extend_empty (Phandle): 包含缺失值的患者数据对象，用于进行插值处理。
    - results_Process_Inter_KNN (result): 存储插值结果和误差指标的对象。
    - output_path (str): 输出文件夹路径，用于保存处理后的结果。

    输出：
    无直接返回值。结果存储在 `results_Process_Inter_KNN` 对象中，处理后的数据以 Excel 文件形式保存到指定路径。

    详细步骤：
    
    """
    
    features = ['id','gender_values', 'age_values', 'iop_od_values','iop_os_values', 'cdr_od_values','cdr_os_values', 'md_od_values', 'md_os_values']
    data_Process_Inter_RandomForest = copy.copy(data_extend)

    data_Process_Inter_RandomForest = Func_Algorithm_random_forest(data_extend_empty, features, n_estimators=100, max_depth=None)
    

    # data_Process_Inter_KNN.cdr_od_values = Func_Algorithm_Inter_KNN(data_extend_empty.cdr_od_values)
    # data_Process_Inter_KNN.cdr_os_values = Func_Algorithm_Inter_KNN(data_extend_empty.cdr_os_values)

    mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_empty.iop_od_values,data_Process_Inter_RandomForest.iop_od_values,data_extend.iop_od_values)
    results_Process_Inter_Random_Forest.iop_od.predicted_list.append(predicted_value)
    results_Process_Inter_Random_Forest.iop_od.true_list.append(true_value)

    mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_empty.iop_os_values,data_Process_Inter_RandomForest.iop_os_values,data_extend.iop_os_values)
    results_Process_Inter_Random_Forest.iop_os.predicted_list.append(predicted_value)
    results_Process_Inter_Random_Forest.iop_os.true_list.append(true_value)



    # path = output_path + '/result_data_extend_Random_Forest_md_process/'
    # Func_output_excel(data_Process_Inter_RandomForest, path)
    
    return results_Process_Inter_Random_Forest
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


def Func_md_Analyse_Evaluate(predicted_series, true_series, absolute_tolerance = 1):
    # 计算均方误差（MSE）
    mse = mean_squared_error(true_series, predicted_series)

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(true_series, predicted_series)

    is_accurate = (
        (predicted_series.between(
            true_series - absolute_tolerance,
            true_series + absolute_tolerance,
            inclusive="both"
        ))
    ).astype(int)
  
    accuracy = is_accurate.mean()

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
        'period_values': data_extend.period_values
    })

    # 输出成excel文件

    os.makedirs(path, exist_ok=True)
    df.to_excel(path + '/patient_' + str(data_extend.ID[0]) + '.xlsx', index=False, header=True)

#眼压 0.15 杯盘比 0.3 视野 0.5 RNFL 0.5

def Func_create_missing_data_multi(data, columns_to_missing, missing_value=666, missing_rate=0.1):
    """
    在指定列引入缺失值，并用 `missing_value` 进行标记
    
    参数：
    - data: pd.DataFrame, 原始数据
    - columns_to_missing: list, 需要引入缺失值的列名
    - missing_value: int/float, 缺失值的标记，默认是 666
    - missing_rate: float, 缺失值的比例（0~1之间），默认 20%

    返回：
    - 带缺失值的数据 DataFrame
    """
    data_missing = copy.copy(data)  # 复制原数据，避免修改原始数据
    
    for col in columns_to_missing:
        if col in data_missing.columns:
            # 随机选择一些行进行缺失
            missing_indices = np.random.choice(
                data_missing.index, 
                size=1, 
                replace=False
            )
            # 用 `missing_value` 替换这些行的数值
            data_missing.loc[missing_indices, col] = missing_value

    return data_missing

def Func_create_missing_data(data):
    data_empty = data
    for attribute in ['iop_od_values', 'iop_os_values', 'cdr_od_values', 'cdr_os_values',
                      'md_od_values', 'md_os_values', 'rnfl_od_values', 'rnfl_os_values']:

        # 获取原始数据的值
        values = getattr(data, attribute)
        if 'iop' in attribute:
            setattr(data_empty, attribute, Func_cmd_1(values, 0.15))
        # if 'cdr' in attribute:
        #     setattr(data_empty, attribute, Func_cmd_1(values, 0.15))
        # if 'md' in attribute:
        #     setattr(data_empty, attribute, Func_cmd_1(values, 0.15))
        # elif 'rnfl' in attribute:
        #     setattr(data_empty, attribute, Func_cmd_1(values, 0.3))
    # 遍历 val 中的列名
    # 将 data_extend 转换为 DataFrame

    # if (len(data_empty.treat_dates)>2):
    #     path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_md_empty_process'
    #     Func_output_excel(data_empty, path)
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
    # num_missing = int(np.ceil(len(non_missing_series) * missing_rate))
    num_missing = 1
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



# 6. 绘制ROC曲线



# 主执行部分
if __name__ == '__main__':
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


    # 指定文件夹路径
    # folder_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data')
    # extend_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_process')
    # extend_empty_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_empty_process')

    # extend_LSTM_train_path =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_md_data\train')
    # # extend_LSTM_test_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\test')
    # extend_LSTM_val_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_md_data\val')

    # extend_GAN_train_path =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_GAN_process\train')
    # extend_GAN_test_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_GAN_process\test')
    # extend_GAN_val_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_GAN_process\val')
    
    # extend_LSTM_path =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_process')
    # extend_LSTM_path_output =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_process_output')
    
    # extend_KNN_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_md_process')
    # extend_KNN_empty_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_md_empty_process')

    extend_randomforest_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\val')
    extend_randomforest_empty_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\test')

    # extend_GAIN_train_path =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\train')
    # extend_LSTM_test_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\test')
    # extend_GAIN_val_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_md_data\val')

    # 加载数据并绘制图表
    # Func_Dataloader2(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_md_data')          #制造缺失数据和对比
    # Func_Process_Inter_KNN(extend_KNN_path, extend_KNN_empty_path)
    # Func_Process_Inter_LSTM(extend_LSTM_train_path,extend_LSTM_val_path)
    # ###Func_Process_Multi_Inter_LSTM(extend_LSTM_train_path, extend_LSTM_test_path,extend_LSTM_val_path)
    # Func_Process_Inter_GAN_Multimodal(extend_GAN_train_path, extend_GAN_test_path,extend_GAN_val_path, gan_model_path=None)
    # Func_Process_Inter_tGAN_Multimodal(extend_GAN_train_path, extend_GAN_test_path, extend_GAN_val_path,tgan_model_path=None)
    # Func_Process_Multi_Inter_GAIN(extend_GAIN_train_path,extend_GAIN_val_path)
    # Func_Process_Multi_Inter_random_forest(extend_randomforest_path,extend_randomforest_empty_path)
    # Func_Inter_iopcdr_LSTM(extend_LSTM_path,extend_LSTM_path_output)
    
    # time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    # output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str
    # os.makedirs(output_path, exist_ok=True)
    # Sample_Feq, the_last_age = Func_origin_data_feq(data)
    # Func_Summary_Feq_plot(output_path,Sample_Feqs, the_last_ages)


