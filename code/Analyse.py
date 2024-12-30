import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime
##数据分析

# def Func_calculate_empty_data_ratio(path):
#     # 遍历文件夹中的所有Excel文件
#     for filename in os.listdir(path):
#         # 确保只处理patient_开头的Excel文件
#         if filename.startswith('patient_') and filename.endswith('.xlsx'):
#             # 完整文件路径
#             file_path = os.path.join(path, filename)
#             try:
#                 # 读取Excel文件
#                 df = pd.read_excel(file_path)
#                 # phandle = Phandle()