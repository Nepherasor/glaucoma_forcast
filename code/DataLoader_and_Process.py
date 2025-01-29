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
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense,Masking,Dropout
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import traceback
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
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



def Func_build_gan_model(input_dim):
    # 构建生成器
    generator = Sequential([
        Dense(256, input_dim=input_dim, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        Dense(input_dim, activation='tanh')  # 输出与输入维度一致
    ], name="Generator")

    # 构建判别器
    discriminator = Sequential([
        Dense(512, input_dim=input_dim, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),  # 防止过拟合
        Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),  # 防止过拟合
        Dense(1, activation='sigmoid')  # 二分类输出
    ], name="Discriminator")
    discriminator.compile(
        optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 构建 GAN
    discriminator.trainable = False
    gan = Sequential([generator, discriminator], name="GAN")
    gan.compile(
        optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
        loss='binary_crossentropy'
    )

    return generator, discriminator, gan


def Func_Process_Inter_GAN_Multimodal(extend_train_path,extend_test_path, extend_val_path, gan_model_path=None):
    """
    使用多模态 GAN 模型填补缺失数据。

    输入：
    - extend_test_path (str): 测试数据路径。
    - extend_val_path (str): 验证数据路径。
    - gan_model_path (str): GAN 模型路径（可选，如果提供则加载预训练模型）。

    输出：
    - Excel 文件，包含 GANs 的预测结果和误差分析。
    """
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = f'E:/BaiduSyncdisk/QZZ/data_generation/output/GAN_Multimodal_{time_str}'
    os.makedirs(output_path, exist_ok=True)

    # 创建输出文件夹
    feature_columns = ['age_values','iop_od_values','iop_os_values','cdr_od_values','cdr_os_values']

    # gan_model_path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code\GAN_inter_model.keras'
    results_Process_Inter_GAN = result()
    # 构建或加载 GAN 模型
    input_dim = len(feature_columns)  # 多模态特征数量：IOP, CDR, MD
    scaler = StandardScaler()
    if gan_model_path:
        generator = load_model(gan_model_path)
    else:
        generator, discriminator, gan = Func_build_gan_model(input_dim)

        data_Process_Inter_GAN_all = []
        #训练部分
        # 遍历文件夹中的所有Excel文件
        for filename in os.listdir(extend_train_path):
            if filename.startswith('patient_') and filename.endswith('.xlsx'):
                file_path = os.path.join(extend_train_path, filename)
                try:
                    # 读取 Excel 文件
                    data_extend = Func_Dataloader_single(file_path)

                    # 提取多模态特征
                    data_Process_Inter_GAN_single = {}
                    for feature in feature_columns:
                        data_Process_Inter_GAN_single[feature] = data_extend.__getattribute__(feature)

                    data_Process_Inter_GAN_all.append(pd.DataFrame(data_Process_Inter_GAN_single))

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

        # 合并所有数据
        GAN_all_data = pd.concat(data_Process_Inter_GAN_all, axis=0, ignore_index=True)

        # 缺失值掩码
        missing_mask = GAN_all_data.isna()

        # 填充 NaN 值为 0（仅用于训练 GANs，不修改原数据）
        combined_data = GAN_all_data.fillna(0)

        # 标准化数据

        GAN_all_data_normalized = pd.DataFrame(
            scaler.fit_transform(combined_data),
            columns=feature_columns
        )

        # 转为 DataFrame
        GAN_all_data_normalized = pd.DataFrame(GAN_all_data_normalized, columns=feature_columns)


        # 训练 GANs
        generator = Func_train_gan_multimodal(generator, discriminator, gan, GAN_all_data_normalized, missing_mask,input_dim)

        generator.save(output_path +'\GAN_inter_model.keras') # 默认保存模型和权重

    # ##预测部分

    for filename in os.listdir(extend_test_path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path_test = os.path.join(extend_test_path, filename)
            file_path_val = os.path.join(extend_val_path, filename)
            try:
                # 加载测试和验证数据
                data_extend_test = Func_Dataloader_single(file_path_test)
                data_extend_test_ori = copy.copy(data_extend_test)
                data_extend_val = Func_Dataloader_single(file_path_val)


                ###缺失部分
                test_column = 'iop_os_values'

                # 提取多模态特征
                data_Process_Inter_GAN_single = {}
                for feature in feature_columns:
                    data_Process_Inter_GAN_single[feature] = data_extend_test.__getattribute__(feature).replace(666,
                                                                                                                np.nan)

                # 转换为 DataFrame
                test_data = pd.DataFrame(data_Process_Inter_GAN_single)

                # 标准化测试数据
                test_data_normalized = pd.DataFrame(
                    scaler.fit_transform(test_data),
                    columns=feature_columns
                )

                # 生成缺失值掩码
                missing_mask = test_data.isna()

                # 用生成器生成缺失值
                noise = np.random.normal(0, 1, size=(missing_mask[test_column].sum(), len(feature_columns)))
                generated_values = generator.predict(noise)

                # 检查生成器输出的形状
                assert generated_values.shape[0] == missing_mask[test_column].sum(), \
                    f"生成器生成的数据数量 ({generated_values.shape[0]}) 与缺失值数量 ({missing_mask[test_column].sum()}) 不一致"

                # 填补缺失值
                filled_data_normalized = test_data_normalized.copy()

                # 将生成器输出展开为一维数组
                generated_values = generated_values.flatten()

                # 按缺失值掩码逐行填充缺失值
                missing_indices = missing_mask.index[missing_mask[test_column]].tolist()
                for idx, value in zip(missing_indices, generated_values):
                    filled_data_normalized.loc[idx, test_column] = value

                # 反标准化结果
                filled_data = pd.DataFrame(
                    scaler.inverse_transform(filled_data_normalized),
                    columns=feature_columns
                )

                # 分析误差
                mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(
                    data_extend_test_ori.iop_os_values,
                    filled_data[test_column],
                    data_extend_val.iop_os_values
                )
                results_Process_Inter_GAN.iop_os.predicted_list.append(predicted_value)
                results_Process_Inter_GAN.iop_os.true_list.append(true_value)


            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()


    # 汇总结果并保存为 Excel

    results_Process_Inter_GAN.iop_os.predicted_val = pd.concat(
            results_Process_Inter_GAN.iop_os.predicted_list, ignore_index=True)
    results_Process_Inter_GAN.iop_os.true_val = pd.concat(
            results_Process_Inter_GAN.iop_os.true_list, ignore_index=True)

    mse, mae, accuracy = Func_Analyse_Evaluate(
            results_Process_Inter_GAN.iop_os.predicted_val,
            results_Process_Inter_GAN.iop_os.true_val
        )

    # 保存结果
    df = pd.DataFrame({
            'predicted_val': results_Process_Inter_GAN.iop_os.predicted_val,
            'true_val': results_Process_Inter_GAN.iop_os.true_val,
            'mse': mse,
            'mae': mae,
            'Acc': accuracy
        })
    df.to_excel(output_path + f'/Analyse_GAN_process.xlsx', index=False, header=True)
    print(mse,mae,accuracy)


def Func_train_gan_multimodal(generator, discriminator, gan, normalized_data, missing_mask,input_dim):
    """
    训练多模态 GAN 模型。

    输入：
    - generator: 生成器模型。
    - discriminator: 判别器模型。
    - gan: GAN 整体模型。
    - data (pd.DataFrame): 标准化后的训练数据。
    - missing_mask (pd.DataFrame): 缺失值掩码。
    - epochs (int): 训练轮数。
    - batch_size (int): 批量大小。

    输出：
    - generator: 训练后的生成器。
    """
    # 提取非缺失行作为真实数据
    # non_missing_data = normalized_data[~missing_mask.any(axis=1)].values
    epochs = 1000
    batch_size = 16


    for epoch in range(epochs):
        # 随机选择一个批次的数据
        batch_indices = np.random.choice(normalized_data.shape[0], batch_size, replace=True)
        real_data = normalized_data.iloc[batch_indices].values
        real_data = normalized_data.iloc[batch_indices].values
        real_mask = missing_mask.iloc[batch_indices].values

        # 判别器训练
        noise = np.random.normal(0, 1, size=(batch_size, input_dim))
        generated_data = generator.predict(noise)

        # 将生成的数据合并到真实数据中，替换缺失部分
        combined_data = real_data.copy()
        combined_data[real_mask] = generated_data[real_mask]

        X_discriminator = np.concatenate([real_data, combined_data], axis=0)
        y_discriminator = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=0)

        # 调试信息
        print(f"X_discriminator shape: {X_discriminator.shape}")
        print(f"y_discriminator shape: {y_discriminator.shape}")

        discriminator.trainable = True
        # 判别器训练

        d_loss, d_acc = discriminator.train_on_batch(X_discriminator, y_discriminator)
        print(f"判别器训练完成: D_loss={d_loss:.4f}, D_acc={d_acc:.4f}")


        # 生成器训练
        noise = np.random.normal(0, 1, size=(batch_size, input_dim))
        y_generator = np.ones((batch_size, 1))  # 欺骗判别器的标签
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_generator)

        # 打印训练信息
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D_loss: {d_loss:.4f}, D_acc: {d_acc:.4f}, G_loss: {g_loss:.4f}")

    return generator



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


# LSTM 构建 TGAN 模型


def Func_build_conditional_tgan_model(input_dim,fill_val,window_size=7):
    """
    构建一个"有条件"的时序GAN，用于缺失插值示例：
      - 生成器输入: (window_size, 2*input_dim)  [ (X_partial, mask) ]
      - 判别器输入: (window_size, 2*input_dim)  [ (X_filled, mask) or (X_real, mask) ]
    返回: (generator, discriminator, gan)
    """

    # ---------- 构建生成器 ----------
    generator = Sequential(name="Generator")
    generator.add(layers.Input(shape=(window_size, 2*input_dim)))
    # 你可以自行决定要不要Masking，若fill_value=666则需自定义处理；很多人会用0表示缺失
    generator.add(layers.Masking(mask_value=fill_val))

    generator.add(layers.LSTM(256, activation='relu', return_sequences=True))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dropout(0.2))

    generator.add(layers.LSTM(512, activation='relu', return_sequences=True))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dropout(0.2))

    # 输出维度: (window_size, input_dim)，即对缺失位置的估计
    generator.add(layers.LSTM(input_dim, activation='tanh', return_sequences=True))

    # ---------- 构建判别器 ----------
    discriminator = Sequential(name="Discriminator")
    discriminator.add(layers.Input(shape=(window_size, 2*input_dim)))
    # 如果仍想Masking就自己改(比如fill_value=666).
    # 但通常Conditional-GAN会把缺失设为0，mask为0/1，就不走Masking层。

    discriminator.add(layers.LSTM(512, activation='relu', return_sequences=True))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dropout(0.3))

    discriminator.add(layers.LSTM(256, activation='relu', return_sequences=False))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dropout(0.3))

    discriminator.add(layers.Dense(1, activation='sigmoid'))

    # 判别器编译
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ---------- 构建GAN: 先freeze判别器 ----------
    discriminator.trainable = False
    gan = Sequential(name="GAN")
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    return generator, discriminator, gan


def Func_train_tgan_conditional(
        generator, discriminator, gan,
        train_data,  # (n_samples, window_size, input_dim), 其中真实缺失用什么表示都行
        real_mask,  # (n_samples, window_size, input_dim), 1=观测, 0=真实缺失
        input_dim,
        batch_size=16,
        epochs=100,
        missing_rate=0.3,  # 对可观测区域额外挖洞比例
        fill_value=0.0  # 人工挖洞后用什么值填
):
    """
    训练条件式 TGAN，既含“真实缺失”又要额外在可观测区域挖洞。
    - train_data: shape (N, T, D)
    - real_mask: shape (N, T, D)  (1=有值, 0=缺失)
    - missing_rate: 在real_mask=1的地方，再随机挖多少
    """
    num_samples = train_data.shape[0]
    window_size = train_data.shape[1]

    for epoch in range(epochs):
        for batch_start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - batch_start)

            X_real = train_data[batch_start: batch_start + current_batch_size].astype(np.float32)
            M_real = real_mask[batch_start: batch_start + current_batch_size].astype(np.float32)

            # =========== 构造额外的 artificial_mask ===========
            # 只在 M_real=1 的地方才可能挖掉
            rnd_mat = np.random.rand(current_batch_size, window_size, input_dim)
            artificial_mask = ((rnd_mat > missing_rate) & (M_real == 1)).astype(np.float32)

            # =========== 融合 ===========
            M_train = (M_real * artificial_mask).astype(np.float32)
            # (1=保留, 0=缺失)

            # =========== 构造 X_partial ===========
            X_partial = M_train * X_real + (1 - M_train) * fill_value

            # =========== 生成器输入 ===========
            G_in = np.concatenate([X_partial, M_train], axis=2)

            # =========== 生成器输出 & 融合 ===========
            X_gen = generator.predict(G_in)
            X_fake = M_train * X_partial + (1 - M_train) * X_gen

            # =========== 判别器输入 ===========
            fake_input = np.concatenate([X_fake, M_train], axis=2)
            # 真数据: 用 (X_real, ???)
            #  - 若你想假设真实数据在训练阶段是完整: (X_real, 1阵)
            #  - 或者 (X_real, M_real)，让判别器也知道真实缺失
            #   => 如果 M_real 里也有 0, 说明那些位置是不可对比
            real_input = np.concatenate([X_real, M_real], axis=2)

            # =========== 判别器标签 ===========
            real_labels = np.ones((current_batch_size, 1), dtype=np.float32)
            fake_labels = np.zeros((current_batch_size, 1), dtype=np.float32)

            # 训练判别器
            d_loss_real = discriminator.train_on_batch(real_input, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_input, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            # 让 fake_input 被判别器误判为真
            G_in = np.concatenate([X_partial, M_train], axis=2)
            g_loss = gan.train_on_batch(G_in, real_labels)

        print(f"[Epoch {epoch + 1}/{epochs}] D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

    return generator, discriminator, gan


def Func_Process_Inter_tGAN_Multimodal(extend_train_path,extend_test_path, extend_val_path, tgan_model_path=None):
    """
    使用多模态 GAN 模型填补缺失数据。

    输入：
    - extend_test_path (str): 测试数据路径。
    - extend_val_path (str): 验证数据路径。
    - gan_model_path (str): GAN 模型路径（可选，如果提供则加载预训练模型）。

    输出：
    - Excel 文件，包含 GANs 的预测结果和误差分析。
    """
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = f'E:/BaiduSyncdisk/QZZ/data_generation/output/GAN_Multimodal_{time_str}'
    os.makedirs(output_path, exist_ok=True)

    # 特征列
    feature_columns = ['gender_values', 'age_values', 'iop_od_values', 'iop_os_values', 'cdr_od_values',
                       'cdr_os_values', 'md_od_values', 'md_os_values', 'rnfl_od_values', 'rnfl_os_values',
                       'period_values']

    # 滑动窗口参数
    window_size = 7  # 每个时间序列的长度为7
    stride = 1  # 每次滑动1步
    FILL_VALUE = 666
    # 创建一个空列表，用于存储用户数据
    tGAN_all_data = []  # 存储未标准化的用户数据
    missing_mask = []   # 存储缺失值掩码

    # 如果有模型路径，则加载模型，否则构建一个新模型
    tgan_model_path = None
    if tgan_model_path:
        generator = load_model(tgan_model_path)
    else:
        generator, discriminator, gan = Func_build_conditional_tgan_model(input_dim=len(feature_columns),fill_val=FILL_VALUE)

        # 读取每个用户的数据（每个文件代表一个用户）
        for filename in os.listdir(extend_train_path):
            if filename.startswith('patient_') and filename.endswith('.xlsx'):
                file_path = os.path.join(extend_train_path, filename)
                try:
                    # 读取 Excel 文件
                    data_extend = Func_Dataloader_single(file_path)

                    # 提取每个用户的多模态特征
                    data_Process_Inter_tGAN_single = {}
                    for feature in feature_columns:
                        data_Process_Inter_tGAN_single[feature] = data_extend.__getattribute__(feature)

                    # 转换为DataFrame
                    user_df = pd.DataFrame(data_Process_Inter_tGAN_single)

                    # 获取用户的缺失数据掩码（True表示缺失，False表示有数据）
                    missing_mask_current = user_df.isna().values
                    missing_mask.append(missing_mask_current)

                    # 用 FILL_VALUE 填充 NaN

                    user_df_filled = user_df.fillna(FILL_VALUE)

                    # 如果用户的数据少于7条，则进行填充
                    user_data_array = user_df_filled.values
                    num_timesteps = user_data_array.shape[0]

                    if num_timesteps < window_size:
                        # 填充数据，使其长度为7
                        padding_size = window_size - num_timesteps
                        padded_user_data_array = np.pad(
                            user_data_array,
                            ((padding_size, 0), (0, 0)),  # 前填充
                            mode='constant',
                            constant_values=FILL_VALUE  # 使用 FILL_VALUE 填充
                        )
                        user_data_array = padded_user_data_array
                        num_timesteps = window_size  # 现在是7条数据

                    # 使用滑动窗口切割大于7条的数据
                    temp_user_data = []
                    for start_idx in range(0, num_timesteps - window_size + 1, stride):
                        # 截取固定长度的时间序列
                        window_data = user_data_array[start_idx:start_idx + window_size]
                        temp_user_data.append(window_data)

                    # 将处理后的数据添加到最终的列表中
                    tGAN_all_data.extend(temp_user_data)

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

        # 将所有用户的处理数据合并成三维数组
        tGAN_all_data = np.array(tGAN_all_data)  # 形状 (n_samples, window_size, n_features)

        # 转为三维数组 (n_samples, window_size, n_features)
        tGAN_all_data = np.array(tGAN_all_data, dtype=float)

        missing_mask_3d = (tGAN_all_data == FILL_VALUE)  # shape: (n_samples, window_size, n_features), dtype=bool
        real_mask_3d = ~missing_mask_3d  # 把缺失取反 => 有值则 True
        real_mask_3d = real_mask_3d.astype(np.float32)

        # =============== 以下为「列级标准化」的实现示例 ===============

        # 1) 获取数据的整体维度
        n_samples, time_steps, n_features = tGAN_all_data.shape

        # 2) 先准备一个副本，以便存放标准化后的结果
        tGAN_all_data_normalized = tGAN_all_data.copy()  # shape 与 tGAN_all_data相同

        # 3) 分列(特征)计算均值和标准差，只基于非填充值
        means = np.zeros(n_features, dtype=float)
        stds  = np.zeros(n_features, dtype=float)

        for f in range(n_features):
            # 取出第 f 列特征在所有 (样本, 时间步) 上的值，排除 FILL_VALUE
            valid_values = tGAN_all_data[:, :, f][tGAN_all_data[:, :, f] != FILL_VALUE]
            if len(valid_values) == 0:
                # 如果这个特征在所有位置都是 FILL_VALUE，那就没法标准化，指定一个默认值
                means[f] = 0.0
                stds[f] = 1.0
            else:
                means[f] = valid_values.mean()
                std_dev  = valid_values.std()
                # 避免除0
                stds[f] = std_dev if std_dev > 1e-12 else 1e-12

        # 4) 应用标准化：对每个特征的有效值做 (x - mean) / std
        for i in range(n_samples):
            for j in range(time_steps):
                for f in range(n_features):
                    if tGAN_all_data_normalized[i, j, f] != FILL_VALUE:
                        tGAN_all_data_normalized[i, j, f] = (
                            (tGAN_all_data_normalized[i, j, f] - means[f]) / stds[f]
                        )
                    else:
                        # 保留填充值不变，以便后续 Masking
                        tGAN_all_data_normalized[i, j, f] = FILL_VALUE

        # =============== 列级标准化到此结束 ====================

        # missing_mask 里 True 表示缺失, False 表示有值
        # 我们要把它转成 real_mask=1 表示有值, 0 表示缺失
        real_mask_current = (~missing_mask_current).astype(np.float32)

        # 填充完后，把 real_mask 做同样的滑窗操作...
        temp_mask_data = []
        for start_idx in range(0, num_timesteps - window_size + 1, stride):
            window_mask = real_mask_current[start_idx: start_idx + window_size]
            temp_mask_data.append(window_mask)

        # 训练 GANs
        generator, discriminator, gan = Func_train_tgan_conditional(
            generator, discriminator, gan,
            tGAN_all_data_normalized, real_mask=real_mask_3d,input_dim=len(feature_columns),
            batch_size=16,
            epochs=100,
            missing_rate=0.3,
            fill_value=FILL_VALUE
        )

        # 保存模型
        generator.save(output_path + '/tGAN_inter_model.keras')

    # ##预测部分

    for filename in os.listdir(extend_test_path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path_test = os.path.join(extend_test_path, filename)
            file_path_val = os.path.join(extend_val_path, filename)
            try:
                # 加载测试和验证数据
                data_extend_test = Func_Dataloader_single(file_path_test)
                data_extend_test_ori = copy.copy(data_extend_test)
                data_extend_val = Func_Dataloader_single(file_path_val)


                ###缺失部分
                test_column = 'iop_os_values'

                # 提取多模态特征
                data_Process_Inter_tGAN_single = {}
                for feature in feature_columns:
                    data_Process_Inter_tGAN_single[feature] = data_extend_test.__getattribute__(feature).replace(666,
                                                                                                                np.nan)

                # 转换为 DataFrame
                test_data = pd.DataFrame(data_Process_Inter_tGAN_single)

                # 标准化测试数据
                test_data_normalized = pd.DataFrame(
                    scaler.fit_transform(test_data),
                    columns=feature_columns
                )

                # 生成缺失值掩码
                missing_mask = test_data.isna()

                # 用生成器生成缺失值
                noise = np.random.normal(0, 1, size=(missing_mask[test_column].sum(), len(feature_columns)))
                generated_values = generator.predict(noise)

                # 检查生成器输出的形状
                assert generated_values.shape[0] == missing_mask[test_column].sum(), \
                    f"生成器生成的数据数量 ({generated_values.shape[0]}) 与缺失值数量 ({missing_mask[test_column].sum()}) 不一致"

                # 填补缺失值
                filled_data_normalized = test_data_normalized.copy()

                # 将生成器输出展开为一维数组
                generated_values = generated_values.flatten()

                # 按缺失值掩码逐行填充缺失值
                missing_indices = missing_mask.index[missing_mask[test_column]].tolist()
                for idx, value in zip(missing_indices, generated_values):
                    filled_data_normalized.loc[idx, test_column] = value

                # 反标准化结果
                filled_data = pd.DataFrame(
                    scaler.inverse_transform(filled_data_normalized),
                    columns=feature_columns
                )

                # 分析误差
                mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(
                    data_extend_test_ori.iop_os_values,
                    filled_data[test_column],
                    data_extend_val.iop_os_values
                )
                results_Process_Inter_tGAN.iop_os.predicted_list.append(predicted_value)
                results_Process_Inter_tGAN.iop_os.true_list.append(true_value)


            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()


    # 汇总结果并保存为 Excel

    results_Process_Inter_tGAN.iop_os.predicted_val = pd.concat(
            results_Process_Inter_tGAN.iop_os.predicted_list, ignore_index=True)
    results_Process_Inter_tGAN.iop_os.true_val = pd.concat(
            results_Process_Inter_tGAN.iop_os.true_list, ignore_index=True)

    mse, mae, accuracy = Func_Analyse_Evaluate(
            results_Process_Inter_tGAN.iop_os.predicted_val,
            results_Process_Inter_tGAN.iop_os.true_val
        )

    # 保存结果
    df = pd.DataFrame({
            'predicted_val': results_Process_Inter_tGAN.iop_os.predicted_val,
            'true_val': results_Process_Inter_tGAN.iop_os.true_val,
            'mse': mse,
            'mae': mae,
            'Acc': accuracy
        })
    df.to_excel(output_path + f'/Analyse_tGAN_process.xlsx', index=False, header=True)
    print(mse,mae,accuracy)


def Func_Process_Inter_LSTM(extend_path,extend_test_path,extend_val_path):
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
    output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str

    results_Process_Inter_LSTM = result()
    data_Process_Inter_LSTM_all = []

    os.makedirs(output_path, exist_ok=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
                data_Process_Inter_LSTM_single['iop_values'] = data_extend.iop_od_values
                data_Process_Inter_LSTM_all.append(pd.DataFrame(data_Process_Inter_LSTM_single))

                data_Process_Inter_LSTM_single = {}
                data_Process_Inter_LSTM_single['ID'] = data_extend.ID * 10 + 2              ##右眼
                data_Process_Inter_LSTM_single['age_values'] = data_extend.age_values
                data_Process_Inter_LSTM_single['iop_values'] = data_extend.iop_os_values
                data_Process_Inter_LSTM_all.append(pd.DataFrame(data_Process_Inter_LSTM_single))



            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


    # model = load_model(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\code\lstm_inter_model.keras')
    LSTM_all_data = pd.concat(data_Process_Inter_LSTM_all, axis=0, ignore_index=True)
    #标准化
    scaler = StandardScaler()
    # 对非缺失值进行标准化
    LSTM_all_data['age_values'] = scaler.fit_transform(LSTM_all_data['age_values'].values.reshape(-1, 1))
    LSTM_all_data['iop_values'] = scaler.fit_transform(LSTM_all_data['iop_values'].values.reshape(-1, 1))


    # 切割
    # 切片数据
    window_size = 3
    X, y = Func_preprocess_and_slice_data(LSTM_all_data, window_size)

    # 确保输入数据为 LSTM 格式 (样本数, 时间步长, 特征数)
    X = X.reshape((X.shape[0], X.shape[1], 2))  # 2个特征 (age 和 iop_values)

    # 创建模型
    model = Func_build_lstm_model(input_shape=(window_size, 2))

    # 训练模型
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=2)
    #
    model.save(output_path +'\lstm_inter_model.keras')  # 默认保存模型和权重



    data_Process_Inter_LSTM_test = []


##读取测试数据，并且把666替换为nan
    for filename in os.listdir(extend_test_path):
        # 确保只处理patient_开头的Excel文件
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            # 完整文件路径
            file_path2 = os.path.join(extend_test_path, filename)
            file_path3 = os.path.join(extend_val_path, filename)
            try:
                # 读取Excel文件
                data_extend_test = Func_Dataloader_single(file_path2)
                data_extend_test_ori = copy.copy(data_extend_test)
                data_extend_val = Func_Dataloader_single(file_path3)

                data_Process_Inter_LSTM_single = pd.DataFrame({
                    'ID': data_extend_test.ID,  # 左眼
                    'age_values': data_extend_test.age_values,
                    'iop_values': data_extend_test.iop_od_values.replace(666, np.nan)  # 替换 666 为 np.nan
                })

                LSTM_test_data = data_Process_Inter_LSTM_single.copy()
                LSTM_test_data['age_values'] = scaler.fit_transform(LSTM_test_data['age_values'].values.reshape(-1, 1))
                LSTM_test_data['iop_values'] = scaler.fit_transform(LSTM_test_data['iop_values'].values.reshape(-1, 1))
                # 预测缺失值
                window_size = 3
                filled_data = Func_LSTM_predict_missing_values(model, LSTM_test_data, window_size, scaler)
                # 3. 填补缺失值后，对结果进行反标准化
                # 将填补后的iop_values列进行反标准化
                filled_data['iop_values'] = Func_reverse_standardization(scaler, filled_data['iop_values'].values)

                # 4. 如果需要，原始的age_values也可以进行反标准化（通常可以直接使用原始的age_values，因为它没有缺失）
                filled_data['age_values'] = data_Process_Inter_LSTM_single['age_values']

                mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_test_ori.iop_od_values,
                                                                                     filled_data['iop_values'],
                                                                                     data_extend_val.iop_od_values)
                results_Process_Inter_LSTM.iop_od.predicted_list.append(predicted_value)
                results_Process_Inter_LSTM.iop_od.true_list.append(true_value)



                data_Process_Inter_LSTM_single = pd.DataFrame({
                    'ID': data_extend_test.ID * 10,  # 右眼
                    'age_values': data_extend_test.age_values,
                    'iop_values': data_extend_test.iop_os_values.replace(666, np.nan)  # 替换 666 为 np.nan
                })



                LSTM_test_data = data_Process_Inter_LSTM_single.copy()
                LSTM_test_data['age_values'] = scaler.fit_transform(LSTM_test_data['age_values'].values.reshape(-1, 1))
                LSTM_test_data['iop_values'] = scaler.fit_transform(LSTM_test_data['iop_values'].values.reshape(-1, 1))
                # 预测缺失值
                filled_data = Func_LSTM_predict_missing_values(model, LSTM_test_data, window_size, scaler)
                # 3. 填补缺失值后，对结果进行反标准化
                # 将填补后的iop_values列进行反标准化
                filled_data['iop_values'] = Func_reverse_standardization(scaler, filled_data['iop_values'].values)

                # 4. 如果需要，原始的age_values也可以进行反标准化（通常可以直接使用原始的age_values，因为它没有缺失）
                filled_data['age_values'] = data_Process_Inter_LSTM_single['age_values']

                mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_test_ori.iop_os_values,
                                                                                     filled_data['iop_values'],
                                                                                     data_extend_val.iop_os_values)
                results_Process_Inter_LSTM.iop_os.predicted_list.append(predicted_value)
                results_Process_Inter_LSTM.iop_os.true_list.append(true_value)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()



    results_Process_Inter_LSTM.iop_od.predicted_val = pd.concat(results_Process_Inter_LSTM.iop_od.predicted_list,ignore_index=True)
    results_Process_Inter_LSTM.iop_od.true_val = pd.concat(results_Process_Inter_LSTM.iop_od.true_list, ignore_index=True)
    mse,mae,accuracy = Func_Analyse_Evaluate(results_Process_Inter_LSTM.iop_od.predicted_val, results_Process_Inter_LSTM.iop_od.true_val)

    # 将 data_extend 转换为 DataFrame
    df = pd.DataFrame({
        'predicted_val': results_Process_Inter_LSTM.iop_od.predicted_val,
        'true_val': results_Process_Inter_LSTM.iop_od.true_val,
        'mse':mse,
        'mae':mae,
        'Acc':accuracy

    })

    # 输出成excel文件
    df.to_excel(output_path + '/Analyse_data_extend_LSTM_process' + '.xlsx', index=False, header=True)





def Func_reverse_standardization(scaler, standardized_values):
    """
    反标准化数据，将标准化后的值还原为原始值。
    """
    return standardized_values * scaler.scale_[0] + scaler.mean_[0]

# 填补缺失值 (通过滑动窗口方式预测)
def Func_LSTM_predict_missing_values(model, test_data, window_size,scaler):
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
    iop_values = test_data['iop_values'].values
    age_values = test_data['age_values'].values

    # 将 666 替换为 NaN


    # 填补所有 NaN 的位置
    # 第一次预测
    for i in range(len(iop_values)):
        if np.isnan(iop_values[i]):  # 找到 NaN 的位置
            # 确保有足够的滑动窗口
            if i >= window_size:
                features = np.column_stack((age_values[i - window_size:i], iop_values[i - window_size:i]))
                features[:, 1] = np.nan_to_num(features[:, 1])  # 替换 NaN 为 0
                features = features.reshape((1, window_size, 2))  # 调整为 LSTM 输入格式

                # 使用模型预测
                predicted = model.predict(features, verbose=0)

                # 检查预测值是否为标量
                if predicted.size == 1:
                    iop_values[i] = predicted.item()  # 提取标量值并填补
                else:
                    print(f"预测值形状异常，位置: {i}, 预测值: {predicted}")

    # 第二次填补：向前/向后填补
    for i in range(len(iop_values)):
        if np.isnan(iop_values[i]):  # 找到剩余的 NaN
            # 尝试向后填补
            j = i + 1
            while j < len(iop_values) and np.isnan(iop_values[j]):
                j += 1
            if j < len(iop_values):  # 找到后面的非空值
                iop_values[i] = iop_values[j]
            else:
                # 如果无法向后填补，尝试向前填补
                j = i - 1
                while j >= 0 and np.isnan(iop_values[j]):
                    j -= 1
                if j >= 0:  # 找到前面的非空值
                    iop_values[i] = iop_values[j]

    # 返回填补后的数据
    test_data['iop_values'] = iop_values
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
        iop_values = group['iop_values'].values
        age_values = group['age_values'].values

        # 检查样本数量是否足够构建窗口
        if len(iop_values) <= window_size:
            continue

        for i in range(len(iop_values) - window_size):
            # 构建特征矩阵 (age_values 和 iop_values)
            features = np.column_stack((age_values[i:i + window_size], iop_values[i:i + window_size]))
            target = iop_values[i + window_size]  # 窗口后面的目标值

            # 跳过包含 NaN 的窗口或目标值
            if np.isnan(features).any() or np.isnan(target):
                continue

            X.append(features)
            y.append(target)

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

    data_Process_Inter_KNN.iop_od_values = Func_Algorithm_Inter_KNN(data_extend_empty.iop_od_values)
    data_Process_Inter_KNN.iop_os_values = Func_Algorithm_Inter_KNN(data_extend_empty.iop_os_values)

    # data_Process_Inter_KNN.cdr_od_values = Func_Algorithm_Inter_KNN(data_extend_empty.cdr_od_values)
    # data_Process_Inter_KNN.cdr_os_values = Func_Algorithm_Inter_KNN(data_extend_empty.cdr_os_values)

    mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_empty.iop_od_values,data_Process_Inter_KNN.iop_od_values,data_extend.iop_od_values)
    results_Process_Inter_KNN.iop_od.predicted_list.append(predicted_value)
    results_Process_Inter_KNN.iop_od.true_list.append(true_value)

    mse, mae, predicted_value, true_value = Func_Analyse_Evaluate_Series(data_extend_empty.iop_os_values,data_Process_Inter_KNN.iop_os_values,data_extend.iop_os_values)
    results_Process_Inter_KNN.iop_os.predicted_list.append(predicted_value)
    results_Process_Inter_KNN.iop_os.true_list.append(true_value)

    path = output_path + '/result_data_extend_KNN_process/'
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
                data_extend_empty = Func_Dataloader_single(file_path2)
                Func_Process_Inter_KNN_single(data_extend, data_extend_empty, results_Process_Inter_KNN, output_path)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    results_Process_Inter_KNN.iop_od.predicted_val = pd.concat(results_Process_Inter_KNN.iop_od.predicted_list,ignore_index=True)
    results_Process_Inter_KNN.iop_od.true_val = pd.concat(results_Process_Inter_KNN.iop_od.true_list, ignore_index=True)
    mse,mae,accuracy = Func_Analyse_Evaluate(results_Process_Inter_KNN.iop_od.predicted_val, results_Process_Inter_KNN.iop_od.true_val)

    # 将 data_extend 转换为 DataFrame
    df = pd.DataFrame({
        'predicted_val': results_Process_Inter_KNN.iop_od.predicted_val,
        'true_val': results_Process_Inter_KNN.iop_od.true_val,
        'mse':mse,
        'mae':mae,
        'Acc':accuracy

    })

    # 输出成excel文件
    df.to_excel(output_path + '/Analyse_data_extend_KNN_process' + '.xlsx', index=False, header=True)



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
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # 指定文件夹路径
    # folder_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data')
    # extend_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_process')
    # extend_empty_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_empty_process')

    # extend_LSTM_train_path =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\train')
    # extend_LSTM_test_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\test')
    # extend_LSTM_val_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_LSTM_process\val')

    extend_GAN_train_path =(r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_GAN_process\train')
    extend_GAN_test_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_GAN_process\test')
    extend_GAN_val_path = (r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata\vaild_data_extend_GAN_process\val')
    # 加载数据并绘制图表
    # Func_Dataloader2(folder_path)          #制造缺失数据和对比
    # Func_Process_Inter_LSTM(extend_LSTM_train_path, extend_LSTM_test_path,extend_LSTM_val_path)
    # Func_Process_Inter_GAN_Multimodal(extend_GAN_train_path, extend_GAN_test_path,extend_GAN_val_path, gan_model_path=None)
    Func_Process_Inter_tGAN_Multimodal(extend_GAN_train_path, extend_GAN_test_path, extend_GAN_val_path,tgan_model_path=None)
    # time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'

    # output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str
    # os.makedirs(output_path, exist_ok=True)
    # Func_Summary_Feq_plot(output_path,Sample_Feqs, the_last_ages)
