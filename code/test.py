import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime


class Phandle:
    def __init__(self):
        # 性别转换：女性为0，男性为1
        self.treat_date = None  # 记录日期
        self.gender_values = None  # 初始化为None

        # 年龄
        self.age_values = None

        # 眼压 (Intraocular Pressure, IOP)
        self.iop_od_values = None  # 右眼眼压
        self.iop_os_values = None  # 左眼眼压

        # 杯盘比 (Cup-to-Disc Ratio, CDR)
        self.cdr_od_values = None  # 右眼杯盘比
        self.cdr_os_values = None  # 左眼杯盘比

        # 视野 (Mean Deviation, MD)
        self.md_od_values = None  # 右眼视野
        self.md_os_values = None  # 左眼视野

        # 视网膜神经纤维层厚度 (Retinal Nerve Fiber Layer, RNFL)
        self.rnfl_od_values = None  # 右眼RNFL
        self.rnfl_os_values = None  # 左眼RNFL
def Func_Dataloader(path):
    """
    遍历文件夹，为每个患者文件绘制图表
    参数:
    path (str): 包含患者Excel文件的文件夹路径
    """
    # 创建输出图表的文件夹（如果不存在）
    output_path = os.path.join(path, 'charts')
    os.makedirs(output_path, exist_ok=True)

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
                # 按照时间排序
                df_sorted = df.sort_values(df.columns[4])
                phandle.treat_dates = np.array(df_sorted.iloc[:, 4].dt.to_pydatetime())

                phandle.gender_values = df_sorted.iloc[:, 1]  # 第十列 性别
                phandle.gender_values = phandle.gender_values.apply(lambda x: 0 if x == '女' else 1)

                phandle.age_values = df_sorted.iloc[:, 3]  # 第十列 年龄

                phandle.iop_od_values = df_sorted.iloc[:, 9]  # 第十列 眼压
                phandle.iop_os_values = df_sorted.iloc[:, 10]  # 第十一列

                phandle.cdr_od_values = df_sorted.iloc[:, 11]  # 第十2列 杯盘比
                phandle.cdr_os_values = df_sorted.iloc[:, 12]  # 第十3列

                phandle.md_od_values = df_sorted.iloc[:, 13]  # 第十2列  视野
                phandle.md_os_values = df_sorted.iloc[:, 14]  # 第十3列

                phandle.rnfl_od_values = df_sorted.iloc[:, 15]  # 第十5列  视野2
                phandle.rnfl_os_values = df_sorted.iloc[:, 16]  # 第十6列

                time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'

                Func_plot(phandle,filename,'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\output')
                # 创建Plotly图表


            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


def Func_plot(data, filename, output_path):
    # 创建包含四个子图的Figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '眼压变化',
            '杯盘比变化',
            '视野变化',
            'RNFL变化'
        )
    )

    # 眼压数据
    fig.add_trace(go.Scatter(
        x=data.treat_dates,
        y=data.iop_od_values,
        mode='markers+lines',
        name='右眼眼压',
        marker=dict(color='blue', size=10)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.treat_dates,
        y=data.iop_os_values,
        mode='markers+lines',
        name='左眼眼压',
        marker=dict(color='red', size=10)
    ), row=1, col=1)

    # 杯盘比数据
    fig.add_trace(go.Scatter(
        x=data.treat_dates,
        y=data.cdr_od_values,
        mode='markers+lines',
        name='右眼杯盘比',
        marker=dict(color='green', size=10)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=data.treat_dates,
        y=data.cdr_os_values,
        mode='markers+lines',
        name='左眼杯盘比',
        marker=dict(color='orange', size=10)
    ), row=1, col=2)

    # 视野数据
    fig.add_trace(go.Scatter(
        x=data.treat_dates,
        y=data.md_od_values,
        mode='markers+lines',
        name='右眼视野',
        marker=dict(color='purple', size=10)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.treat_dates,
        y=data.md_os_values,
        mode='markers+lines',
        name='左眼视野',
        marker=dict(color='brown', size=10)
    ), row=2, col=1)

    # RNFL数据
    fig.add_trace(go.Scatter(
        x=data.treat_dates,
        y=data.rnfl_od_values,
        mode='markers+lines',
        name='右眼RNFL',
        marker=dict(color='pink', size=10)
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=data.treat_dates,
        y=data.rnfl_os_values,
        mode='markers+lines',
        name='左眼RNFL',
        marker=dict(color='cyan', size=10)
    ), row=2, col=2)

    # 更新布局
    fig.update_layout(
        height=800,
        width=1600,
        title_text=f'患者 {filename.replace("patient_", "").replace(".xlsx", "")} 的治疗数据',
        template='plotly_white',
        showlegend=True
    )

    # 更新x、y轴标签
    fig.update_xaxes(title_text='治疗日期', row=1, col=1)
    fig.update_yaxes(title_text='眼压值', row=1, col=1)

    fig.update_xaxes(title_text='治疗日期', row=1, col=2)
    fig.update_yaxes(title_text='杯盘比值', row=1, col=2)

    fig.update_xaxes(title_text='治疗日期', row=2, col=1)
    fig.update_yaxes(title_text='视野指标值', row=2, col=1)

    fig.update_xaxes(title_text='治疗日期', row=2, col=2)
    fig.update_yaxes(title_text='RNFL厚度值', row=2, col=2)

    # 保存交互式HTML文件
    output_file = os.path.join(output_path, f'{filename.replace(".xlsx", ".html")}')
    pyo.plot(fig, filename=output_file, auto_open=False)
    print(f"成功为 {filename} 生成图表: {output_file}")

# 主执行部分
if __name__ == '__main__':
    # 指定文件夹路径
    folder_path = r'E:\BaiduSyncdisk\QZZ\data_generation\data_generation\divdata'
    # 加载数据并绘制图表
    Func_Dataloader(folder_path)