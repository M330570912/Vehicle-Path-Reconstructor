# coding: utf-8
# 训练一个Node2Vec模型 输出路网各个路段拓扑结构的embedding向量（32维或64维）


#输入数据文件来源
#AVI路段流量文件：inputs\\AVI_Links_Flow\\{date}_{st}-{et}.npy
#FCD路段流量文件：outputs\\FCD_Links_Flow\\{date}_{st}-{et}.npy

import numpy as np
import pandas as pd
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarDiGraph
from gensim.models import Word2Vec
import Helper as helper
from sklearn.manifold import TSNE
import sklearn.pipeline as pl # 数据管线
from sklearn.preprocessing import StandardScaler # 输入数据的标准化
import matplotlib.pyplot as plt

#Step 1:准备工作 建立路网有向图
# 读取路口与路段数据
# 1.1 读取路段ID、长度、限速
links_IDs,links_length,links_speed_limit = helper.Get_Links_Info("inputs\\路段.csv")
# 1.2 读取路口ID(str)、路口经度、路口纬度
cross_ids,lon_ls,lat_ls = helper.Get_Nodes_Info("inputs\\节点.csv")

# 1.3 定义算法全局变量
walk_length = 10 # 每个随机游走样本的最大长度（路径长度）
n = 10  # 每个节点进行随机游走的样本数量（路径数量）
p = 1  
q = 10
vector_size = 32 #输出的向量维数
epochs = 200 #训练迭代次数

# 1.4 准备节点的信息：lon经度 lat纬度 index节点的名称（本例直接按数字编号，缺331 333 两个点）
node_data = pd.DataFrame({"lon": lon_ls, "lat": lat_ls}, index = cross_ids)
# 1.5 准备路段的信息：起点 终点 权重（取路段长度的倒数）FCD路流量 限速 通行时间 车道数
weighted_edge_data = pd.DataFrame(
    {
        "source": [str(int(x[0])) for x in links_IDs],
        "target": [str(int(x[1])) for x in links_IDs],
        "weight": [1. for _ in links_IDs] #暂时是均等权重
    }
)

# 1.6 使用节点与路段信息，构造一个有向图路网
G = StellarDiGraph({"corner": node_data}, {"line": weighted_edge_data})
# # 1.7 查看有向图的信息
# print(G.info())
# # 1.8 简要查看各路段权重的分布
# _, weights = G.edges(include_edge_weight=True)
# plt.figure(figsize=(6, 4))
# plt.title("Edge weights histogram")
# plt.ylabel("Count")
# plt.xlabel("edge weights")
# plt.hist(weights,100)
# plt.show()

#Step 2:建立Node2Vec模型
# 2.1 通过有向图建立随机游走对象
rw = BiasedRandomWalk(G)
# 2.2 进行随机游走采样
weighted_walks = rw.run(
    nodes = G.nodes(),  # root nodes
    length = walk_length,  # maximum length of a random walk
    n = n,  # number of random walks per root node
    p = p,  # Defines (unormalised) probability, 1/p, of returning to source node
    q = q,  # Defines (unormalised) probability, 1/q, for moving away from source node
    weighted=True,  # for weighted random walks
    seed = 42,  # random seed fixed for reproducibility
)
print("随机游走样本数: {}".format(len(weighted_walks)))

# 2.3 使用Word2Vec内核构造Node2Vec模型，并进行训练
model_Node2Vec = Word2Vec(
    weighted_walks, vector_size=vector_size, window=5, min_count=0, sg=1, workers=8, epochs = epochs
)

#Step 3:读取Node2Vec模型输出的结果
# 3.1 读取节点embedding向量字典 {路口ID：向量array}
node_embeddings = {}
for item in cross_ids:
    node_embeddings[item] = model_Node2Vec.wv[item]

# 3.2 再根据某种规则 由节点向量构造出边向量字典 {路段ID：向量array}
edge_embeddings = {}
for key in links_IDs:
    o = str(int(key[0]))
    d = str(int(key[1]))

    #加权平均构造边向量
    edge_embeddings[key] = (np.mean([node_embeddings[o],node_embeddings[d]],0))

    # #直接拼接构造边向量
    # edge_embeddings[key] = (np.array(list(node_embeddings[o])+list(node_embeddings[d])))

#保存路段的embedding向量字典
# np.save("outputs\\Edge_Embeddings_Vectors\\"+ "edge_embeddings_{}_{}_{}_{}_{}_{}".format(walk_length,n,p,q,vector_size,epochs) +".npy",edge_embeddings)
np.save("outputs\\Edge_Embeddings_Vectors\\edge_embeddings.npy",edge_embeddings)