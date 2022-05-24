# coding: utf-8
# 训练一个Autoencoder模型 输出路网各个路段特征属性的embedding向量（32维或64维） 
# X(FCD路段流量，路段长度，路段限速，路段平均行程时间，车道数)

#输入数据文件来源
#FCD路段流量文件：outputs\\FCD_Links_Flow\\{date}_{st}-{et}.npy
import numpy as np
import tensorflow as tf
from keras import layers, losses
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.metrics import mean_absolute_error



#用自编码器去训练一个路段属性的向量表示
# 加载数据集 处理一下 归一化
def Autoencoder_Input(features):
  feature_0 = [x[0]/150. for x in features] #除以150
  feature_1 = [x[1]/1500. for x in features] #除以1500
  feature_2 = [x[2]/50. for x in features] #除以50
  feature_3 = [x[3]/150. for x in features] #除以150
  feature_4 = [x[4]/10. for x in features] #除以10

  edge_feature = [(feature_0[i],feature_1[i],feature_2[i],feature_3[i],feature_4[i]) for i in range(len(features))]
  edge_feature = np.array(edge_feature)
  return edge_feature

# 处理一下自编码器输出的结果 反归一化 （可视化用的）
def Autoencoder_Output(decoded_feature):
  decoded_feature_0 = [x[0]*150. for x in decoded_feature] #乘以150
  decoded_feature_1 = [x[1]*1500. for x in decoded_feature] #乘以1500
  decoded_feature_2 = [x[2]*50. for x in decoded_feature] #乘以50
  decoded_feature_3 = [x[3]*150. for x in decoded_feature] #乘以150
  decoded_feature_4 = [x[4]*10. for x in decoded_feature] #乘以10

  last_decoded_edge_feature = [(decoded_feature_0[i],decoded_feature_1[i],decoded_feature_2[i],decoded_feature_3[i],decoded_feature_4[i]) for i in range(len(decoded_feature))]
  last_decoded_edge_feature = np.array(last_decoded_edge_feature)
  return last_decoded_edge_feature



edge_feature = Autoencoder_Input(links_train_X)


x_train,x_test = train_test_split(edge_feature,test_size=0.2,random_state=42)


#中间层的维度
latent_dim = vector_size


class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(5)),
      layers.Dense(latent_dim, activation='relu',
      activity_regularizer=regularizers.l1(10e-5)
      ),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(5, activation='sigmoid'),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


autoencoder = Autoencoder(latent_dim)

# x_train使用作为输入和目标来训练模型。将encoder学习将数据集从 5 维映射到潜在空间，decoder并将学习重建原始数据。
autoencoder.compile(optimizer='adam', loss=losses.MeanAbsoluteError())

autoencoder.fit(x_train, x_train,
                epochs=50,
                shuffle=True,
                validation_data=(x_test, x_test))


# 现在模型已经训练好了，让我们通过对测试集中的图像进行编码和解码来测试它。
encoded_edge_feature = autoencoder.encoder(x_test).numpy()
decoded_edge_feature = autoencoder.decoder(encoded_edge_feature).numpy()

# 结果展示
# 处理一下 反归一化
last_decoded_edge_feature = Autoencoder_Output(decoded_edge_feature)

#评估一下测试集误差

index = 0
mae = mean_absolute_error(x_test[:,index], last_decoded_edge_feature[:,index])
print(mae)

# 现在自编码器模型已经训练好了，直接用来导出所有路段的edge feature
# 获取编码结果
edge_feature_all = autoencoder.encoder(Autoencoder_Input(links_predict_X)).numpy()