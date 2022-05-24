# coding: utf-8
import numpy as np
import Helper as helper
import matplotlib.pyplot as plt

#Step 1:准备工作

#1.1 指定时间段
datetime = "20210106_10-11" #"20210106_6-7"

##1.2 读取路网信息：路口位置表，路段ID表，路段属性表，路口拓扑结构表（连通性表），最短路表，可行路径表
#1.2.1读取路口位置信息
cross_ids,cross_locations = helper.Load_Nodes("inputs\\节点.csv")

#1.2.2读取路段信息表 #2021.08.01改动 删除links_ID 没用
links_info = helper.Load_Links("inputs\\路段.csv") 

#1.2.3计算路网拓扑结构表，生成路网拓扑表字典 格式：路网ID：相邻路口ID
map_info = helper.Calc_Map_Info(cross_ids,links_info)

#1.2.4路网相关数据准备：路段的通行能力 路口的最大流量
links_capacity = np.load('inputs\\links_capacity.npy',allow_pickle=True).item() #路段的通行能力 为0的是缺少信息的，后期算法里用2400为缺省值
max_node_volumn = np.load('inputs\\max_node_volumn.npy',allow_pickle=True).item() #路口的最大流量



#直接读取预生成的任意两点之间最短路字典（10层的） 避免浪费过多时间
shortest_paths = np.load('inputs\\shortest_paths_10.npy',allow_pickle=True).item() 


#1.2.6读取进口方向——上游路口ID映射表 格式：{(路口ID,进口方向):对应的路段ID} #2021.08.01新增
Direction_Map = helper.Load_Direction_Map("inputs\\Direction_Map.csv")





#2.2 这个是PFE要用到的输入 路网中所有路段的流量值
links_flow_all = np.load('outputs\\All_Links_Flow_Before_PFE\\' + datetime + '.npy',allow_pickle=True).item()


#2.3 这个是PFE要用到的输入 FCD的载客路径集
FCD_paths_all = np.load('inputs\\20210106_20210120_载客路径集.npy',allow_pickle=True).item()
#处理一下，筛选出早上7.00-8.00的路径集（工作日的）
FCD_paths_set = []
for date in FCD_paths_all.keys():
    #判断工作日
    day = date[:4]
    #周末的数据就跳过
    if day in ["0109","0110","0116","0117"]:
        continue
    #判断时间 
    if ("0700-" not in date) and ("0800-" not in date):
        continue 
    data = FCD_paths_all[date]
    for item in data.values():
        temp = []
        for node in item:
            temp.append(node[0])
        temp = tuple(temp)
        if len(temp) >= 2 and temp not in FCD_paths_set:
            FCD_paths_set.append(temp)



#Step 3:路径重构算法：PFE方法

print("开始进行路径流量估计算法。")
#3.1.2 路径重构主函数B:选用PFE方法： #输入：不完整的车辆路径（用来读取各OD对之间的需求）、粒子滤波方法生成的车辆路径(用来补充算法所需路径集)、AVI观测路段的流量、路段信息表、路段通行能力表
OD_Paths_Result_PFE,links_flow_PFE,obj_ls,MAE_links,MAE_paths = helper.Generate_PFE_Paths(FCD_paths_set,links_flow_all,links_info,links_capacity)
print("路径流量估计算法已完成。")
#保存算法输出结果
np.save("outputs\\All_Paths_Flow_After_PFE\\"+ datetime +".npy", OD_Paths_Result_PFE)
np.save("outputs\\All_Links_Flow_After_PFE\\"+ datetime +".npy", links_flow_PFE)

#输出算法误差随迭代次数的变化
print("PFE_Links_MAE_values:",MAE_links)
print("PFE_Paths_MAE_values:",MAE_paths)


#3.2绘图准备：PFE误差分析
plt.figure(figsize=(7,7))
fig = plt.figure(1)
plt.subplots_adjust(wspace=0.3) #调整路段图的内间距
fig.canvas.set_window_title('PFE Result')


#评估对比两者的差异 输入重构的路段流量、真实的路段流量（字典）
links_flow_PFE_for_error_calc = {}
for key in links_flow_all.keys():
    links_flow_PFE_for_error_calc[key] = links_flow_PFE.get(key,links_flow_all[key])
    if links_flow_PFE_for_error_calc[key] == 0:
        links_flow_PFE_for_error_calc[key] = links_flow_all[key]
        
#简单画图分析一下误差数据
helper.Evaluate_Links(plt.subplot(111),links_flow_all,links_flow_PFE_for_error_calc,lim=3000)

#显示图像
plt.show()