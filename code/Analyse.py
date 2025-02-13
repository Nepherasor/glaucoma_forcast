import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
##数据分析

def Func_read_data_result(path):
    # 遍历文件夹中的所有Excel文件

        df = pd.read_excel(path)
        # 跳过表头,读取第一列和第二列数据
        pred_values = np.array(df.iloc[1:, 0])  # 第一列为预测值
        true_values = np.array(df.iloc[1:, 1])  # 第二列为真实值
        
        
        return true_values, pred_values
                # phandle = Phandle()


def calculate_roc_metrics(true_values, pred_values, threshold=0.1):
    """
    计算ROC曲线相关指标
    
    参数:
        true_values: 真实值数组
        pred_values: 预测值数组 
        threshold: 阈值,默认0.1
        
    返回:
        fpr: 假阳性率
        tpr: 真阳性率
        roc_auc: ROC曲线下面积
    """
    # 创建二分类标签
    pred_probabilities = (np.abs(true_values - pred_values) <= threshold * true_values).astype(int)
    y_true_binary = pred_probabilities

    # 计算ROC曲线指标
    fpr, tpr, thresholds = roc_curve(y_true_binary, pred_values)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def plot_roc_curve(fpr_KNN, tpr_KNN, roc_auc_KNN, fpr_LSTM, tpr_LSTM, roc_auc_LSTM, fpr_GAN, tpr_GAN, roc_auc_GAN, fpr_TGAN, tpr_TGAN, roc_auc_TGAN,
                   colors={'roc': 'darkorange', 'baseline': 'darkgray'},
                   labels={'x': 'False Positive Rate',
                          'y': 'True Positive Rate',
                          'title': 'Receiver Operating Characteristic',
                          'roc': 'ROC curve (area = {:.2f})'},
                   line_width=2):
    """
    使用plotly绘制ROC曲线
    
    参数:
        fpr: 假阳性率
        tpr: 真阳性率 
        roc_auc: ROC曲线下面积
        colors: 颜色字典
        labels: 标签字典
        line_width: 线宽
    """
    import plotly.graph_objects as go
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:\BaiduSyncdisk\QZZ\data_generation\output' + '/' + time_str
    
    # 创建1行4列的子图
    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=('KNN ROC Curve', 'LSTM ROC Curve', 
                                      'GAN ROC Curve', 'TGAN ROC Curve'))
    


    # 添加KNN ROC曲线
    fig.add_trace(go.Scatter(x=fpr_KNN, y=tpr_KNN, name=f'ROC (AUC={roc_auc_KNN:.2f})',
                            line=dict(color=colors['roc'], width=line_width)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Baseline',
                            line=dict(color=colors['baseline'], width=line_width, dash='dash')), row=1, col=1)
                            
    # 添加LSTM ROC曲线
    fig.add_trace(go.Scatter(x=fpr_LSTM, y=tpr_LSTM, name=f'ROC (AUC={roc_auc_LSTM:.2f})',
                            line=dict(color=colors['roc'], width=line_width)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Baseline',
                            line=dict(color=colors['baseline'], width=line_width, dash='dash')), row=1, col=2)
                            
    # 添加GAN ROC曲线
    fig.add_trace(go.Scatter(x=fpr_GAN, y=tpr_GAN, name=f'ROC (AUC={roc_auc_GAN:.2f})',
                            line=dict(color=colors['roc'], width=line_width)), row=1, col=3)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Baseline',
                            line=dict(color=colors['baseline'], width=line_width, dash='dash')), row=1, col=3)
                            
    # 添加TGAN ROC曲线
    fig.add_trace(go.Scatter(x=fpr_TGAN, y=tpr_TGAN, name=f'ROC (AUC={roc_auc_TGAN:.2f})',
                            line=dict(color=colors['roc'], width=line_width)), row=1, col=4)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Baseline',
                            line=dict(color=colors['baseline'], width=line_width, dash='dash')), row=1, col=4)

    # 更新布局
    fig.update_layout(
        height=400,
        width=1600,
        showlegend=False,
        title_text='ROC Curves Comparison'
    )
    
    # 更新所有子图的坐标轴范围和标签
    for i in range(1, 5):
        fig.update_xaxes(title_text=labels['x'], range=[0, 1], row=1, col=i)
        fig.update_yaxes(title_text=labels['y'], range=[0, 1.05], row=1, col=i)

    # 保存交互式HTML文件
    output_file = 'roc_curves_comparison.html'
    fig.write_html(output_file)
    print(f"成功保存ROC曲线对比图表到: {output_file}")
    fig.show()

# 调用函数绘制图表
    # 读取所有数据
path_KNN = r'C:\Users\Nephe\Desktop\Analyse_data_extend_KNN_process.xlsx'
path_LSTM = r'C:\Users\Nephe\Desktop\Analyse_data_extend_LSTM_process.xlsx'
path_GAN = r'C:\Users\Nephe\Desktop\Analyse_data_extend_GAN_process.xlsx'
path_TGAN = r'C:\Users\Nephe\Desktop\Analyse_data_extend_cdr_LSTM_process.xlsx'

    # KNN
true_values_KNN, pred_values_KNN = Func_read_data_result(path_KNN)
fpr_KNN, tpr_KNN, roc_auc_KNN = calculate_roc_metrics(true_values_KNN, pred_values_KNN, threshold=0.1)
    
    # LSTM
true_values_LSTM, pred_values_LSTM = Func_read_data_result(path_LSTM)
fpr_LSTM, tpr_LSTM, roc_auc_LSTM = calculate_roc_metrics(true_values_LSTM, pred_values_LSTM, threshold=0.1)
    
    # GAN
true_values_GAN, pred_values_GAN = Func_read_data_result(path_GAN)
fpr_GAN, tpr_GAN, roc_auc_GAN = calculate_roc_metrics(true_values_GAN, pred_values_GAN, threshold=0.1)
    
    # TGAN
true_values_TGAN, pred_values_TGAN = Func_read_data_result(path_TGAN)
fpr_TGAN, tpr_TGAN, roc_auc_TGAN = calculate_roc_metrics(true_values_TGAN, pred_values_TGAN, threshold=0.1)
plot_roc_curve(fpr_KNN, tpr_KNN, roc_auc_KNN, fpr_LSTM, tpr_LSTM, roc_auc_LSTM, fpr_GAN, tpr_GAN, roc_auc_GAN, fpr_TGAN, tpr_TGAN, roc_auc_TGAN)

