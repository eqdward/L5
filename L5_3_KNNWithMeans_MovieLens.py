# -*- coding: utf-8 -*-
"""
@author: yy
"""

from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import KFold
from surprise import accuracy

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(r'C:\Users\yy\Desktop\BI\L5\L5-1\L5-code\knn_cf\ratings.csv', reader=reader)

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})

kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):   
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # 计算RMSE和MAE
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)
