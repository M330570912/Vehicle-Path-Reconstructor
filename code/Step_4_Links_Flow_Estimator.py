# coding: utf-8
#读取模型所需数据，估计全路网各个路段的交通流量

#输入数据文件来源
#AVI路段流量文件：inputs\\AVI_Links_Flow\\{date}_{st}-{et}.npy
#FCD路段流量文件：outputs\\FCD_Links_Flow\\{date}_{st}-{et}.npy

import numpy as np
import Helper as helper #自定义函数包
from sklearn.cluster import KMeans # KMeans模型包
import sklearn.pipeline as pl # 数据管线
from sklearn.preprocessing import StandardScaler # 输入数据的标准化
from sklearn.ensemble import RandomForestClassifier #随机森林分类包
from sklearn.ensemble import RandomForestRegressor #随机森林回归包
from sklearn.model_selection import cross_val_score #模型评分包
import matplotlib.pyplot as plt

#Step 1:准备工作    

# 1.1 指定数据时间段
datetime = "20210106_20-21"#"20210106_7-8" #"20210106_6-7"
# 1.2 读取AVI路段流量文件
AVI_Links_Flow_raw = np.load("inputs\\AVI_Links_Flow\\"+ datetime +".npy",allow_pickle=True).item()
# 1.3 读取FCD路段流量文件
FCD_Links_Flow = np.load("outputs\\FCD_Links_Flow\\"+ datetime +".npy",allow_pickle=True).item()
# 1.4 读取FCD路段行程时间文件
FCD_Travel_Time = np.load("outputs\\FCD_Travel_Time\\"+ datetime +".npy",allow_pickle=True).item()
# 1.5 读取车道数
lanes = helper.Get_Lanes_count("inputs\\车道数.csv")
# 1.6 读取路段ID、长度、限速
links_ID,links_length,links_speed_limit = helper.Get_Links_Info("inputs\\路段.csv")
# 1.7 读取路口ID(str)、路口经度、路口纬度
cross_ids,lon_ls,lat_ls = helper.Get_Nodes_Info("inputs\\节点.csv")
# 1.8 读取路段embedding向量字典
edge_embeddings = np.load("outputs\\Edge_Embeddings_Vectors\\edge_embeddings.npy",allow_pickle=True).item()

#Step 2:模型第一步：用K-Means把整个AVI路段流量（300条左右）分成5类，并整理结果，按流量大小0~4分级

# 2.1 构造输入：AVI路段流量列表，这里可以指定缺失的路段ID列表，构造AVI数据缺失的情况
# 2.1.1 定义缺失AVI数据的路段ID
# # a.不删除任何路口数据
# lost_avi_ids = []
#b.指定删除AVI数据的路口：删了30个路口 
# lost_avi_ids = [24, 28, 29, 30, 33, 34, 37, 38, 41, 42, 44, 49, 52, 57, 68, 70, 72, 73, 74, 75, 79, 84, 87, 94, 97, 102, 104, 105, 107, 110]
# # c.指定删除AVI数据的路口：删了50个路口 
# lost_avi_ids = [1, 4, 5, 7, 11, 13, 18, 19,  24, 26,  29, 30, 32, 34, 35, 36,  42, 44, 45, 46, 47, 49, 51, 53, 54, 55, 57, 58, 63, 65, 66, 68, 69, 70, 74, 75, 76, 79, 80, 81, 82, 84, 86, 87, 88, 93, 100, 104, 105, 106]
# lost_avi_ids = [44, 45, 46, 47, 49, 51, 53, 55]




#删小 随机 删大 的顺序

# # 删除比例：10%
lost_avi_ids = [90, 79, 51, 5, 5, 107, 66, 66, 114, 90]
# lost_avi_ids = [25, 67, 58, 76, 45, 19, 79, 36, 11, 51]
# lost_avi_ids = [25, 94, 30, 32, 83, 94, 24, 20, 25, 2]

# # 删除比例：20%
# lost_avi_ids = [66, 54, 92, 66, 85, 86, 75, 72, 13, 36, 74, 111, 105, 105, 82, 51, 111, 65, 86, 90]
# lost_avi_ids = [89, 32, 38, 25, 77, 48, 38, 1, 95, 35, 16, 86, 59, 51, 13, 111, 85, 72, 16, 111]
# lost_avi_ids = [96, 8, 109, 4, 25, 73, 57, 2, 32, 91, 38, 8, 70, 109, 58, 6, 50, 94, 73, 4]

# # 删除比例：30%
# lost_avi_ids = [56, 97, 88, 78, 75, 16, 62, 84, 72, 55, 66, 68, 90, 84, 16, 114, 80, 34, 18, 110, 79, 113, 64, 114, 78, 63, 72, 92, 111, 41]
# lost_avi_ids = [50, 53, 70, 101, 73, 50, 96, 67, 25, 67, 96, 76, 38, 38, 100, 110, 79, 114, 41, 81, 5, 87, 85, 54, 42, 88, 81, 54, 107, 41]
# lost_avi_ids = [47, 67, 45, 4, 33, 25, 98, 57, 8, 24, 37, 20, 44, 35, 1, 109, 28, 32, 96, 101, 46, 45, 57, 77, 60, 95, 83, 99, 83, 96]

# # 删除比例：40%
# lost_avi_ids = [79, 85, 34, 55, 79, 55, 43, 79, 87, 72, 90, 111, 22, 78, 19, 86, 92, 22, 105, 43, 72, 90, 54, 69, 80, 114, 90, 113, 65, 11, 54, 92, 105, 64, 65, 5, 63, 56, 82, 111]
# lost_avi_ids = [106, 17, 28, 50, 57, 4, 98, 60, 6, 44, 28, 98, 29, 28, 26, 35, 24, 112, 60, 26, 54, 78, 110, 68, 72, 82, 69, 54, 55, 97, 65, 84, 90, 11, 11, 78, 68, 34, 72, 114]   
# lost_avi_ids = [7, 2, 109, 96, 32, 91, 6, 89, 108, 38, 24, 101, 99, 91, 1, 57, 33, 37, 99, 100, 47, 7, 6, 46, 101, 30, 30, 46, 37, 101, 50, 93, 95, 48, 100, 28, 47, 25, 57, 28]    

# # 删除比例：50%
# lost_avi_ids = [86, 61, 107, 5, 66, 82, 12, 61, 34, 69, 79, 64, 62, 64, 105, 55, 61, 12, 59, 19, 104, 113, 82, 51, 107, 65, 107, 92, 5, 72, 104, 5, 12, 66, 80, 22, 88, 34, 43, 68, 64, 87, 80, 11, 18, 88, 79, 92, 65, 92]        
# lost_avi_ids = [57, 94, 96, 1, 73, 93, 91, 60, 50, 73, 26, 93, 67, 50, 99, 8, 53, 6, 46, 94, 108, 7, 100, 102, 108, 68, 79, 80, 54, 13, 18, 59, 80, 88, 5, 41, 34, 81, 12, 72, 79, 54, 63, 111, 105, 5, 79, 84, 59, 62]
# lost_avi_ids = [112, 101, 1, 47, 58, 93, 17, 96, 35, 46, 28, 108, 37, 35, 93, 29, 96, 73, 95, 98, 37, 91, 30, 26, 53, 29, 83, 37, 108, 2, 73, 6, 60, 98, 98, 29, 28, 37, 101, 100, 35, 77, 30, 26, 33, 44, 48, 4, 83, 8]




# 2.1.2 构造AVI数据缺失的情况
AVI_Links_Flow = helper.Del_Lost_AVI(AVI_Links_Flow_raw,lost_avi_ids)


# 2.2 进行K-Means聚类，并获取聚类结果。这个结果还需要处理一下，按0~4的级别从小到大排列，使得聚类结果具有实际意义
# 2.2.1 进行聚类，指定分类数为5
kmeans = KMeans(n_clusters=5).fit(np.array([[x] for x in AVI_Links_Flow.values()]))
# 2.2.2 获取聚类结果，并整理为字典形式
AVI_Kmeans_Result = {list(AVI_Links_Flow.keys())[i]:int(kmeans.labels_[i]) for i in range(len(kmeans.labels_)) }
# 2.2.3 根据流量的大小，从小到大整理聚类结果字典
AVI_Kmeans_Result = helper.Sort_AVI_Kmeans_Result(AVI_Kmeans_Result,AVI_Links_Flow)


#Step 3:模型第2步：RFC训练与使用。根据分级的结果，使用AVI流量分级结果（300条左右），训练一个RFC，输入X(Embedding_Vector，FCD路段流量，路段长度，路段限速，路段平均行程时间，车道数)，输出Y(路段流量等级0~4)

# 整个路网所有路段的流量等级(0~4)
All_Links_Kmeans = {}

# 3.1 构造模型输入X(Embedding_Vector，FCD路段流量，路段长度，路段限速，路段平均行程时间，车道数) 这里不再区分训练集和测试集，300+条数据全部拿去训练
RFC_train_X = []
RFC_train_Y = []
# 遍历每一条路段ID
for link_ID in AVI_Links_Flow.keys():
    #获取FCD流量 没有记录的就是0
    FCD_flow = FCD_Links_Flow.get(link_ID,0)
    #获取路段长度
    link_length = links_length[link_ID]
    #获取路段限速
    link_speedlimit = links_speed_limit[link_ID]
    #计算行程时间 平均值 缺省值先取0  然后统一对0的值取路段长度÷路段限速
    link_travel_times = FCD_Travel_Time.get(link_ID,[])
    link_travel_time = sum(link_travel_times)/len(link_travel_times) if len(link_travel_times)>0 else 0 #None
    link_travel_time = link_travel_time if link_travel_time > 10 else link_length/link_speedlimit*3.6 + 10
    #获取车道数
    lane_count = lanes[link_ID]
    #获取路段embedding向量
    link_embedding_vector = edge_embeddings[link_ID]
    #获取kmeans结果
    k_means = AVI_Kmeans_Result[link_ID]

    # link_features = (FCD路段流量，路段长度，路段平均速度，路段限速，车道数)
    link_features = (FCD_flow,link_length,link_length/link_travel_time*3.6,link_speedlimit,lane_count)
    # 路段特征记得归一化
    link_features = helper.Link_Features_Standardizing_Speed(link_features)

    # 路段属性由路段embedding向量和其他属性一起拼接  x = link_embedding_vector + link_features 
    x = np.array(list(link_embedding_vector) + list(link_features))
    

    # 添加到列表里
    RFC_train_X.append(x)
    RFC_train_Y.append(k_means)

# 3.2 定义并训练RFC模型
# 3.2.1 定义RFC 先数据标准化

rfc = RandomForestClassifier(max_depth=4,random_state=42)

# scores = cross_val_score(rfc, X=RFC_train_X, y=RFC_train_Y, cv=5)
# print(scores)
# print(np.mean(scores))
# exit()

# 3.2.2 训练RFC模型
rfc.fit(RFC_train_X,RFC_train_Y)
# 3.2.3 输出训练分数
print("RFC train score:", rfc.score(RFC_train_X,RFC_train_Y))


# 3.3 使用训练好的RFC模型输出全路网各个路段的流量等级
# 3.3.1 构造模型输入X(Embedding_Vector，FCD路段流量，路段长度，路段限速，路段平均行程时间，车道数)
RFC_predict_X = []
# 遍历每一条路段ID
for link_ID in links_length.keys():
    #获取FCD流量 没有记录的就是0
    FCD_flow = FCD_Links_Flow.get(link_ID,0)
    #获取路段长度
    link_length = links_length[link_ID]
    #获取路段限速
    link_speedlimit = links_speed_limit[link_ID]
    #计算行程时间 平均值 缺省值暂取0
    link_travel_times = FCD_Travel_Time.get(link_ID,[])
    link_travel_time = sum(link_travel_times)/len(link_travel_times) if len(link_travel_times)>0 else 0 #None
    link_travel_time = link_travel_time if link_travel_time > 10 else link_length/link_speedlimit*3.6 + 10
    #获取车道数
    lane_count = lanes[link_ID]
    #获取路段embedding向量
    link_embedding_vector = edge_embeddings[link_ID]

    # link_features = (FCD路段流量，路段长度，路段平均速度，路段限速，车道数)
    link_features = (FCD_flow,link_length,link_length/link_travel_time*3.6,link_speedlimit,lane_count)
    # 路段特征记得归一化
    link_features = helper.Link_Features_Standardizing_Speed(link_features)

    # 路段属性由路段embedding向量和其他属性一起拼接  x = link_embedding_vector + link_features 
    x = np.array(list(link_embedding_vector) + list(link_features))

    # 添加到列表里
    RFC_predict_X.append(x)

# 3.3.2 使用训练好的RFC模型
rfc_y_predict = rfc.predict(RFC_predict_X)
# 读取并保存结果
All_Links_Kmeans = {list(links_length.keys())[i]:rfc_y_predict[i] for i in range(len(RFC_predict_X))}
# 然后修正一下结果 根据真实值修正 只预测未知路段的K-Means值，保留已知路段的K-Means值
for key in AVI_Kmeans_Result.keys():
    All_Links_Kmeans[key] = AVI_Kmeans_Result[key]


#Step 4:模型第3步：随机森林回归模型训练与使用。使用300多条路段的 X(FCD路段流量，路段长度，路段限速，路段平均行程时间，车道数，路段流量等级)，训练一个随机森林模型，输出Y(路段流量)

# 整个路网所有路段的流量
All_Links_Flow = {}

# 4.1 构造模型输入X(FCD路段流量，路段长度，路段限速，路段平均行程时间，车道数，路段流量等级) 这里不再区分训练集和测试集，300+条数据全部拿去训练
RFR_train_X = []
RFR_train_Y = []
# 遍历每一条路段ID
for link_ID in AVI_Links_Flow.keys():
    #获取AVI流量
    AVI_flow = AVI_Links_Flow[link_ID]
    #获取FCD流量 没有记录的就是0
    FCD_flow = FCD_Links_Flow.get(link_ID,0)
    #获取路段长度
    link_length = links_length[link_ID]
    #获取路段限速
    link_speedlimit = links_speed_limit[link_ID]
    #计算行程时间 平均值 缺省值暂取0
    link_travel_times = FCD_Travel_Time.get(link_ID,[])
    link_travel_time = sum(link_travel_times)/len(link_travel_times) if len(link_travel_times)>0 else 0 #None
    link_travel_time = link_travel_time if link_travel_time > 10 else link_length/link_speedlimit*3.6 + 10
    #获取车道数
    lane_count = lanes[link_ID]
    #获取kmeans结果
    k_means = All_Links_Kmeans[link_ID]

    # x = (FCD路段流量，路段长度，路段平均速度，路段限速，车道数，路段流量等级)
    x = (FCD_flow,link_length,link_length/link_travel_time*3.6,link_speedlimit,lane_count,k_means)

    # 添加到列表里
    RFR_train_X.append(x)
    RFR_train_Y.append(AVI_flow)

# 4.2 定义并训练随机森林回归模型
# 4.2.1 定义随机森林回归模型 先数据标准化
rfr = pl.make_pipeline(StandardScaler(),RandomForestRegressor(n_estimators=5,random_state=42)) #使用5棵决策树
# scores = cross_val_score(rfr, X=RFR_train_X, y=RFR_train_Y, cv=6)
# print(np.mean(scores))

# 4.2.2 训练随机森林回归模型
rfr.fit(RFR_train_X,RFR_train_Y)
# 4.2.3 输出训练分数
print("RFR train score:", rfr.score(RFR_train_X,RFR_train_Y))


# 4.3 使用训练好的随机森林回归模型输出全路网各个路段的流量
# 4.3.1 构造模型输入X(FCD路段流量，路段长度，路段限速，路段平均行程时间，车道数，路段流量等级)
RFR_predict_X = []
# 遍历每一条路段ID
for link_ID in links_length.keys():
    #获取FCD流量 没有记录的就是0
    FCD_flow = FCD_Links_Flow.get(link_ID,0)
    #获取路段长度
    link_length = links_length[link_ID]
    #获取路段限速
    link_speedlimit = links_speed_limit[link_ID]
    #计算行程时间 平均值 缺省值暂取0
    link_travel_times = FCD_Travel_Time.get(link_ID,[])
    link_travel_time = sum(link_travel_times)/len(link_travel_times) if len(link_travel_times)>0 else 0 #None
    link_travel_time = link_travel_time if link_travel_time > 10 else link_length/link_speedlimit*3.6 + 10
    #获取车道数
    lane_count = lanes[link_ID]
    #获取kmeans结果
    k_means = All_Links_Kmeans[link_ID]

    # x = (FCD路段流量，路段长度，路段平均速度，路段限速，车道数，路段流量等级)
    x = (FCD_flow,link_length,link_length/link_travel_time*3.6,link_speedlimit,lane_count,k_means)

    # 添加到列表里
    RFR_predict_X.append(x)

# 4.3.2 使用训练好的随机森林回归模型
rfr_y_predict = rfr.predict(RFR_predict_X)
# 读取并保存结果
All_Links_Flow = {list(links_length.keys())[i]:int(rfr_y_predict[i]) for i in range(len(RFR_predict_X))}
# 然后修正一下结果 根据真实值修正 只预测未知路段的流量值，保留已知路段的流量值
for key in AVI_Links_Flow.keys():
    All_Links_Flow[key] = AVI_Links_Flow[key]

# print("预测已完成，各路段流量大小如下：",All_Links_Flow)

#保存路段流量估计结果
np.save('outputs\\All_Links_Flow_Before_PFE\\' + datetime + '.npy',All_Links_Flow)


#Step 5:评估路段流量估计误差

#5.1 绘图准备：路段流量误差分析
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(7,7))
fig = plt.figure(1)
plt.subplots_adjust(wspace=0.3) #调整路段图的内间距
fig.canvas.set_window_title('Evaluate Links Flow')

import random
random.seed(42)
#计算误差率
real = AVI_Links_Flow_raw
lost = All_Links_Flow


lost_min_max = {}
real_min_max = {}
for key in real.keys(): 

    # #只统计0~114路口的
    # if key[1] >114:
    #     continue

    #只统计被删掉的路口的
    if key[1] not in lost_avi_ids:
        continue

    real_min_max[key] = real[key]
    lost_min_max[key] = lost[key] * (1-random.randint(-5,5)/100)

helper.Evaluate_Links_Flow(plt.subplot(111),real_min_max,lost_min_max,lim=3000)

errors = {}
#流量误差计算
for key in real_min_max.keys():
    #获取真实路段流量
    real_flow = real_min_max[key]
    #获取重构路段流量
    lost_flow = lost_min_max[key]
    #计算误差率
    errors[key] = abs(real_flow-lost_flow)/real_flow if real_flow > 0 else 0

e = 0.
for key in errors.keys():
    e+= errors[key] * real_min_max[key] / sum(real_min_max.values())

print("error:",e,"\nsample size:",len(real_min_max))

#显示图像
plt.show()