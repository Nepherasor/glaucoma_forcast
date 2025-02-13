import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py_offline
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime


class Phandle:
    def __init__(self):
        self.treat_dates = None  # 记录日期
        self.gender_values = None  # 性别
        self.age_values = None  # 年龄
        self.iop_od_values = None  # 右眼眼压
        self.iop_os_values = None  # 左眼眼压
        self.cdr_od_values = None  # 右眼杯盘比
        self.cdr_os_values = None  # 左眼杯盘比
        self.md_od_values = None  # 右眼视野
        self.md_os_values = None  # 左眼视野
        self.rnfl_od_values = None  # 右眼RNFL
        self.rnfl_os_values = None  # 左眼RNFL


def Func_Dataloader(path):
    time_str = datetime.now().strftime('%Y%m%d_%H%M') + '/'
    output_path = 'E:/BaiduSyncdisk/QZZ/data_generation/data_generation/output' + '/' + time_str
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(path):
        if filename.startswith('patient_') and filename.endswith('.xlsx'):
            file_path = os.path.join(path, filename)
            try:
                df = pd.read_excel(file_path)
                df_sorted = df.sort_values(df.columns[4])
                phandle = Phandle()
                phandle.treat_dates = np.array(df_sorted.iloc[:, 4].dt.to_pydatetime())
                phandle.gender_values = df_sorted.iloc[:, 1].apply(lambda x: 0 if x == '女' else 1)
                phandle.age_values = df_sorted.iloc[:, 3]
                phandle.iop_od_values = df_sorted.iloc[:, 9]
                phandle.iop_os_values = df_sorted.iloc[:, 10]
                phandle.cdr_od_values = df_sorted.iloc[:, 11]
                phandle.cdr_os_values = df_sorted.iloc[:, 12]
                phandle.md_od_values = df_sorted.iloc[:, 13]
                phandle.md_os_values = df_sorted.iloc[:, 14]
                phandle.rnfl_od_values = df_sorted.iloc[:, 15]
                phandle.rnfl_os_values = df_sorted.iloc[:, 16]
                Func_plot(phandle, filename, output_path)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


def Func_plot(data, filename, output_path):
    fig = make_subplots(rows=2, cols=2, subplot_titles=('眼压变化', '杯盘比变化', '视野变化', 'RNFL变化'))

    # 添加眼压数据
    fig.add_trace(go.Scatter(x=data.treat_dates, y=data.iop_od_values, mode='markers+lines', name='右眼眼压'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=data.treat_dates, y=data.iop_os_values, mode='markers+lines', name='左眼眼压'), row=1,
                  col=1)

    # 添加杯盘比数据
    fig.add_trace(go.Scatter(x=data.treat_dates, y=data.cdr_od_values, mode='markers+lines', name='右眼杯盘比'), row=1,
                  col=2)
    fig.add_trace(go.Scatter(x=data.treat_dates, y=data.cdr_os_values, mode='markers+lines', name='左眼杯盘比'), row=1,
                  col=2)

    # 添加视野数据
    fig.add_trace(go.Scatter(x=data.treat_dates, y=data.md_od_values, mode='markers+lines', name='右眼视野'), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=data.treat_dates, y=data.md_os_values, mode='markers+lines', name='左眼视野'), row=2,
                  col=1)

    # 添加RNFL数据
    fig.add_trace(go.Scatter(x=data.treat_dates, y=data.rnfl_od_values, mode='markers+lines', name='右眼RNFL'), row=2,
                  col=2)
    fig.add_trace(go.Scatter(x=data.treat_dates, y=data.rnfl_os_values, mode='markers+lines', name='左眼RNFL'), row=2,
                  col=2)

    fig.update_layout(height=800, width=1600,
                      title_text=f'患者 {filename.replace("patient_", "").replace(".xlsx", "")} 的治疗数据',
                      template='plotly_white', showlegend=True)
    fig.update_xaxes(title_text='治疗日期', row=1, col=1)
    fig.update_yaxes(title_text='眼压值', row=1, col=1)
    fig.update_xaxes(title_text='治疗日期', row=1, col=2)
    fig.update_yaxes(title_text='杯盘比值', row=1, col=2)
    fig.update_xaxes(title_text='治疗日期', row=2, col=1)
    fig.update_yaxes(title_text='视野指标值', row=2, col=1)
    fig.update_xaxes(title_text='治疗日期', row=2, col=2)
    fig.update_yaxes(title_text='RNFL厚度值', row=2, col=2)

    output_file = os.path.join(output_path, f'{filename.replace(".xlsx", ".html")}')
    py_offline.plot(fig, filename=output_file, auto_open=False)
    print(f"成功为 {filename} 生成图表: {output_file}")


if __name__ == '__main__':
    folder_path = r'E:/BaiduSyncdisk/QZZ/data_generation/data_generation/divdata/md4'
    Func_Dataloader(folder_path)