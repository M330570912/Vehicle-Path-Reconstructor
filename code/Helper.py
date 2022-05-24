# coding: utf-8
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random
from pandas.core.frame import DataFrame

# 1.4 读取车道数数据
def Get_Lanes_count(fileName):
    lanes_pd = pd.read_csv(fileName,index_col=None)
    lanes = {}
    for _,row in lanes_pd.iterrows():
        lane_o = int(row["起点"])
        lane_d = int(row["终点"])
        lane_count = int(row["车道数"])
        lanes[(lane_o,lane_d)] = lane_count
    return lanes

#读取路段ID、长度、限速数据
def Get_Links_Info(fileName):
    links_ID = []
    links_length = {}
    links_speed_limit = {}
    links_pd = pd.read_csv(fileName, index_col = False)
    for _,row in links_pd.iterrows():

        s_id = int(row["上游路口ID"]) # ssn起点编号（范围1-369）
        e_id = int(row["下游路口ID"]) # esn终点编号
        od = (s_id,e_id) #直接用这个od来代替路段的ID 便于算法使用
        length = float(row["路段长度"]) # len长度
        speedlimit = int(row["限速"]) # speedlimit限速 km/h

        links_ID.append(od)
        links_length[od] = length
        links_speed_limit[od] = speedlimit

    return links_ID,links_length,links_speed_limit


#读取路口节点信息  路口ID(str化) 路口经度  路口纬度
def Get_Nodes_Info(fileName):
    cross_ids = []
    lon_ls = []
    lat_ls = []
    datas = pd.read_csv(fileName, index_col = False)
    for _,row in datas.iterrows():
        x = row["经度"] #经度
        y = row["纬度"] #纬度
        cross_id = row["路口ID"] #路口ID

        cross_ids.append(str(int(cross_id)))
        lon_ls.append(x)
        lat_ls.append(y)
    
    return cross_ids,lon_ls,lat_ls

#删掉指定路口的AVI路段流量数据
def Del_Lost_AVI(avi_raw,lost_avi_ids):
    new_avi_data = {}
    for key in avi_raw.keys():
        if key[1] in lost_avi_ids:
            continue
        new_avi_data[key] = avi_raw[key]

    return new_avi_data

#仅保留指定路口的AVI路段流量数据
def Save_Lost_AVI(avi_raw,lost_avi_ids):
    new_avi_data = {}
    for key in avi_raw.keys():
        if key[1] not in lost_avi_ids:
            continue
        new_avi_data[key] = avi_raw[key]

    return new_avi_data


#整理Kmeans聚类结果 按流量从低到高排序 0 1 2 3 4 
def Sort_AVI_Kmeans_Result(kmeans_result,avi_links_flow):

    temp = {}
    #先遍历所有分类结果，从0~4类各找到一个具体的流量数据 如{0:155, 1:899, 2:322, 3:28, 4:2433} 这是原始的聚类结果，不一定是按流量大小排序的 我们需要整理一下
    for sorted_id in np.unique(list(kmeans_result.values())):
        for link_id in kmeans_result.keys():
            sort = kmeans_result[link_id]
            value = avi_links_flow[link_id]
            if sort == sorted_id:
                temp[value] = sorted_id
                break

    
    #从低到高排序
    values = list(temp.keys())
    values.sort()
    #找到对应的分类
    new_sort = {}
    for i in range(len(values)):
        new_sort[i] = temp[values[i]]
    
    new_sort = {b:a for a,b in new_sort.items()}

    #然后重新修改分类
    for key in kmeans_result.keys():
        kmeans_result[key] = new_sort[kmeans_result[key]]
    
    return kmeans_result

#路段属性的归一化 #(FCD路段流量，路段长度，路段限速，路段平均行程时间，车道数)
def Link_Features_Standardizing(features):
  feature_0 = features[0]/150. #除以150
  feature_1 = features[1]/1500. #除以1500
  feature_2 = features[2]/50.  #除以100
  feature_3 = features[3]/150. #除以150
  feature_4 = features[4]/10.  #除以10

  edge_features = np.array([feature_0,feature_1,feature_2,feature_3,feature_4])

  return edge_features

#路段属性的归一化 #(FCD路段流量，路段长度，路段平均速度，路段限速，车道数)
def Link_Features_Standardizing_Speed(features):
  feature_0 = features[0]/150. #除以150
  feature_1 = features[1]/1500. #除以1500
  feature_2 = features[2]/100. #除以100
  feature_3 = features[3]/50.  #除以100
  feature_4 = features[4]/10.  #除以10

  edge_features = np.array([feature_0,feature_1,feature_2,feature_3,feature_4])

  return edge_features

#评估函数 评估两组路径的路段流量差异
def Evaluate_Links_Flow(ax, real_path_link_flow, re_path_link_flow,lim=3000):
    #1.统计各个路段的流量 画图  横轴真实流量 纵轴重构流量 
    real_path_link_flow = list(real_path_link_flow.values())
    re_path_link_flow = list(re_path_link_flow.values())
    ax.set_xlim(xmax=lim,xmin=0) #横轴刻度
    ax.set_ylim(ymax=lim,ymin=0) #纵轴刻度
    # ax.set_xlabel('Real Link Flow (veh)')
    # ax.set_ylabel('Reconstruction Link Flow (veh)')
    # ax.set_title("Links Flow")
    ax.set_xlabel('真实路段流量（辆/小时）')
    ax.set_ylabel('重构路段流量（辆/小时）')
    ax.set_title("路段流量对比")
    ax.scatter(real_path_link_flow, re_path_link_flow,s=5,c="red",zorder=10)
    ax.plot([0,lim], [0,lim], linewidth=1,zorder=5)
    #计算 MAE RMSE R2
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    import math
    mae = mean_absolute_error(real_path_link_flow, re_path_link_flow)
    rmse = math.sqrt(mean_squared_error(real_path_link_flow, re_path_link_flow))

    #标注mae,rmse
    ax.text(lim*0.1, lim*0.9, f"MAE: {mae:0.1f}, RMSE: {rmse:0.1f}", fontsize=10) 
    return mae,rmse


#读取数据文件
def Load_Target_Datas(database_file,start_time,end_time,margin=3000): #默认余量±30分钟
    #根据start_time和end_time判断是否已经存在数据文件，存在了就不需要再提取了
    target_datas_filename = str(start_time)[4:-2] + "-" + str(end_time)[4:-2] + ".csv"

    if os.path.exists(target_datas_filename):
        target_datas = pd.read_csv(target_datas_filename, index_col = False)
        return target_datas
    else:
        #读取大数据库数据
        datas = pd.read_csv(database_file, index_col = False)

        #根据给出的时间节点筛选需要生成路径的数据

        #筛选数据：初步筛选 按照余量 往前推30分钟 往后推30分钟
        target_datas = datas[(datas["时间"] > start_time - margin) & (datas["时间"] < end_time + margin)]
        #按车牌 时间 升序排序
        target_datas = target_datas.sort_values(by=["车牌","时间"],inplace=False,ascending=[True,True])
        #储存数据文件
        target_datas.to_csv(target_datas_filename,index=False,encoding="utf-8")
        #删除缓存
        del datas
        del target_datas

        target_datas = pd.read_csv(target_datas_filename, index_col = False)

        return target_datas
    


#显示数据量 有效路口数量 缺失的路口编号 #2021.12.21更新 缺失路口数按全部367个路口统计 （除了331、333这两个不存在的路口）
def Show_Datas_Info(datas_pd):
    datas_count = len(datas_pd) #数据量
    veh_count = len(np.unique(datas_pd["车牌"])) #车辆数
    cross_id = np.unique(datas_pd["路口ID"]) #路口ID编号
    cross_count = len(cross_id) #路口数量
    lost_rate = (1-cross_count/367) * 100 #数据缺失率 #以前是除以114
    missing_cross_id_str = "" #缺失的路口ID
    missing_cross_id = []
    for i in range(1,368):
        if i == 331 or i == 333:
            continue
        if i not in cross_id:
            missing_cross_id_str += (str(i) + " ")
            missing_cross_id.append(i)

    
    return "数据量：{}, 车辆数：{}, 路口数量：{}, 缺失率：{:.1f}%, 缺失的路口：[{}]".format(datas_count,veh_count,cross_count,lost_rate,missing_cross_id_str), missing_cross_id,list(cross_id)


#计算邻接矩阵字典 格式：路网ID：相邻路口ID
def Calc_Map_Info(cross_ids,links_info):
    map_info = {}
    #遍历每个路口ID
    for cross_id in cross_ids:
        surrounding_ls = []
        #遍历每个路段信息 查找看看有没有起点是这个路段的
        for link_info in list(links_info.values()):
            s_id = link_info["起点路口ID"]
            e_id = link_info["终点路口ID"]
            if cross_id == s_id: #如果这条路段的起点就是给定的路口
                surrounding_ls.append(e_id) #那就把路段终点添加到周围路口的列表里
        map_info[cross_id] = surrounding_ls #遍历完成后记录下来周围路口
    
    return map_info

#读取路段信息
def Load_Links(fileName):
    links_pd = pd.read_csv(fileName, index_col = False)
    links_info = {}
    #links_ID = {} #记录每个路段的id与OD
    for _,row in links_pd.iterrows():
        info = {}
        link_id = int(row["路段ID"]) #路段ID
        s_id = int(row["上游路口ID"]) # ssn起点编号（范围1-369）
        e_id = int(row["下游路口ID"]) # esn终点编号
        od = (s_id,e_id) #直接用这个od来代替路段的ID 便于算法使用
        length = float(row["路段长度"]) # len长度
        lanes = int(row["车道数"]) # lanes车道数
        speedlimit = int(row["限速"]) # speedlimit限速 km/h
        # ramp = int(row["ramp"]) # ramp是否有匝道 0无1上2下匝道
        # linktype = int(row["mylinktype"]) # mylinktype路段类型1主2次3支路

        info["编号"] = link_id
        info["起点路口ID"] = s_id
        info["终点路口ID"] = e_id
        info["长度"] = length
        info["车道数"] = lanes
        info["限速"] = speedlimit
        # info["匝道类型"] = ramp
        # info["路段类型"] = linktype
        info["OD"] = od

        links_info[od] = info
        #links_ID[link_id] = od

    return links_info #,links_ID


#读取路口节点信息
def Load_Nodes(fileName):
    cross_ids = []

    cross_locations = {} #路口位置经纬度字典 {路口ID:[x,y]} x经度 y纬度
    datas = pd.read_csv(fileName, index_col = False)
    for _,row in datas.iterrows():
        x = row["经度"] #经度
        y = row["纬度"] #纬度
        cross_id = row["路口ID"] #路口ID

        cross_ids.append(int(cross_id))
        cross_locations[cross_id] = [x,y] #添加到位置字典里
    
    return cross_ids,cross_locations


#构造OD补全数据集 2021.12.21更新 使用FCD数据 统计当前路网缺失条件下，FCD车辆被检测到的O、D 和 真实的O、D
def Generate_FCD_OD_Map(FCD_avi_OD:dict,available_cross_IDs:list,mode="O"): #mode为 "O" 或 "D"
    FCD_avi_extend = {}
    #遍历每一辆FCD的路径
    for path in FCD_avi_OD.values():
        #区分 O D两种模式 D的话就把path倒序一下
        if mode=="O":
            path = path
        elif mode == "D":
            path.reverse()
        else:
            return None

        #看看FCD的路径中 哪个路口是第一个被检测到的
        for i in range(len(path)):
            if path[i] not in available_cross_IDs:
                continue
            #直到出现第一个 把这个点之前的片段都截下来
            partial = path[:i]
            #合规性检验：这个partial里不能出现有AVI设备的点
            flag = True
            for cross in partial:
                if cross in available_cross_IDs:
                    #说明这个路径片段无效 或者部分无效？先全部标记为无效 看看效果咋样
                    flag = False
                    break
            if flag:
                #然后添加到数据集里
                FCD_avi_extend[path[i]] = FCD_avi_extend.get(path[i],[])
                if len(partial) == 0:
                    break
                #再区分一下 O D两种模式 D模式要倒序回来
                if mode=="D":
                    partial.reverse()
                #已有的可以重复添加，频数越大，说明这条路径概率越大
                FCD_avi_extend[path[i]].append(partial)
            break

    return FCD_avi_extend

#2021.12.21更新：考虑FCD数据补充的AVI原始OD补全
#根据AVI数据表 生成原始车辆路径（未经过最短路修补） min_len表示车辆路径最少需要包含几个数据点
def Generate_Origin_Paths(AVI_Datas,FCD_avi_O_extend:dict,FCD_avi_D_extend:dict,min_len=1, max_len=99):
    min_len = 1 if min_len < 1 else min_len
    paths = {} #存储车辆路径 车辆ID：[路段1,路段2,...]
    paths_with_time ={} #这个数据是包含时间的 车辆ID：[[路段1,路段2,...],[时间1,时间2,...]]

    #因为数据表里的数据已经按照车牌和时间升序排列了
    #我们只要按行读取处理就行
    path = []
    path_time = []
    pre_veh_id = "No1" #上一行的车牌
    pre_cross_id = -1 #上一行的路口ID

    for _,row in AVI_Datas.iterrows(): #遍历每行
        veh_id = str(row["车牌"]) #读取车牌
        cross_id = int(row["路口ID"]) #读取路口ID
        time = int(row["时间"]) #读取时间
        #如果这行的车牌和上一行的车牌不同 那就相当于出现了新车 需要先保存上一个车辆的信息 然后清空临时信息表
        if veh_id != pre_veh_id:
            if len(path) >= min_len and len(path) <= max_len: #如果path列表不是空的 且路径长度小于max_len（个别错误数据会使得路径数很长 应该是车牌错了） 那就保存上一个车辆的信息到字典中
                
                paths[pre_veh_id] = path #记录到字典中
                paths_with_time[pre_veh_id] = [path,path_time] #记录到字典中

            #只要这一行的车辆ID和上一行的不同 那么就把上一行的数据清空（不管有没有保存，因为没有保存的都是长度不符合条件的）
            path = [] #然后清空临时信息表
            path_time = [] #然后清空临时信息表
            pre_cross_id = -1 #重置上一行的路口ID

        #如果这一行的路口ID和上一行的是一样的 那就忽略这一行的数据 避免连续两次出行相同的路口
        if cross_id != pre_cross_id:
            path.append(cross_id)#添加到临时列表中
            path_time.append(time) #添加到临时列表中

        pre_veh_id = veh_id #记录上一行的车牌号
        pre_cross_id = cross_id #记录上一行的路口号

    #2021.12.21新增：考虑FCD数据补充的AVI原始OD补全
    #遍历每一辆车
    for key in paths.keys():
        #获取原始路径
        path = paths[key]
        #获取原始路径的时间
        path_time = paths_with_time[key]
        #获取O、D
        O = path[0]
        D = path[-1]
        #先对O点进行延伸修补
        #找到修补O点的候选集
        O_candidates = FCD_avi_O_extend.get(O,[])
        #如果候选集不为空 那就生成1个随机数 任选一个补进去
        if len(O_candidates) !=0:
            r = random.choice(O_candidates)
            #然后添加到path前边
            path = r + path
            #同时处理一下path_time
            path_time[0] = r + path_time[0]
            #时间也要增加相同数量的进去
            for _ in r:
                path_time[1].insert(0,path_time[1][0])
        
        #然后对D点进行延伸修补
        #找到D点的候选集
        D_candidates = FCD_avi_D_extend.get(D,[])
        #如果候选集不为空 那就生成1个随机数 任选一个补进去
        if len(D_candidates) !=0:
            r = random.choice(D_candidates)
            #然后添加到path后边
            path = path + r
            #同时处理一下path_time
            path_time[0] = path_time[0] + r
            #时间也要增加相同数量的进去
            for _ in r:
                path_time[1].append(path_time[1][-1])
        
        #最后别忘了保存到字典里
        paths[key] = path
        paths_with_time[key] = path_time


    print("原始车辆路径已生成，共{}条。".format(len(paths)))
    return paths,paths_with_time

# #根据AVI数据表 生成原始车辆路径（未经过最短路修补） min_len表示车辆路径最少需要包含几个数据点
# def Generate_Origin_Paths(AVI_Datas:DataFrame,Direction_Map:dict,min_len=1, max_len=21):
#     min_len = 1 if min_len < 1 else min_len
#     paths = {} #存储车辆路径 车辆ID：[路段1,路段2,...]
#     paths_with_time ={} #这个数据是包含时间的 车辆ID：[[路段1,路段2,...],[时间1,时间2,...]]
    
#     #2021.09.01更新：车辆路径考虑车辆在各个路口的驶入方向（以前只看路口ID，现在要通过路口ID+驶入方向倒推车辆来自哪个路口）
#     AVI_Datas["真实路口ID"] = AVI_Datas.apply(lambda x: Direction_Map[(x.路口ID,x.进口方向)][0],axis=1)

#     #因为数据表里的数据已经按照车牌和时间升序排列了
#     #我们只要按行读取处理就行
#     path = []
#     path_time = []
#     pre_veh_id = "No1" #上一行的车牌
#     pre_cross_id = -1 #上一行的路口ID
#     pre_real_cross_id = -1 #上一行的真实路口ID

#     for _,row in AVI_Datas.iterrows(): #遍历每行
#         veh_id = str(row["车牌"]) #读取车牌
#         cross_id = int(row["路口ID"]) #读取路口ID
#         real_cross_id = int(row["真实路口ID"]) #读取路口ID
#         time = int(row["时间"]) #读取时间
#         #如果这行的车牌和上一行的车牌不同 那就相当于出现了新车 需要先保存上一个车辆的信息 然后清空临时信息表
#         if veh_id != pre_veh_id:
#             if len(path) >= min_len and len(path) <= max_len: #如果path列表不是空的 且路径长度小于max_len（个别错误数据会使得路径数很长 应该是车牌错了） 那就保存上一个车辆的信息到字典中
                
#                 path.append(pre_cross_id)
#                 paths[pre_veh_id] = path #记录到字典中

#                 path_time.insert(0,path_time[0]) #第一个路口的经过时刻取重复值
#                 paths_with_time[pre_veh_id] = [path,path_time] #记录到字典中

#             #只要这一行的车辆ID和上一行的不同 那么就把上一行的数据清空（不管有没有保存，因为没有保存的都是长度不符合条件的）
#             path = [] #然后清空临时信息表
#             path_time = [] #然后清空临时信息表
#             pre_cross_id = -1 #重置上一行的路口ID
#             pre_real_cross_id = -1 #重置上一行的真实路口ID

#         #如果这一行的路口ID和上一行的是一样的 那就忽略这一行的数据 避免连续两次出现相同的路口
#         if real_cross_id == pre_real_cross_id:
#             continue
#         path.append(real_cross_id)#添加到临时列表中
#         path_time.append(time) #添加到临时列表中

#         pre_veh_id = veh_id #记录上一行的车牌号
#         pre_cross_id = cross_id #记录上一行的路口号
#         pre_real_cross_id = real_cross_id #记录上一行的真实路口号




#     print("原始车辆路径已生成，共{}条。".format(len(paths)))
#     return paths,paths_with_time


#搜索给定OD之间的最短路
def OD_Shortest_Paths(o,d,map_info,search_times):
    shortest_path = []
    all_paths = []
    o_surroundings = map_info[o] #获取O点相邻的路口
    #如果D点就在O点相邻的位置 那么直接返回就行
    if d in o_surroundings:
        print("找到了{}-{}之间的最短路。".format(o,d))
        return [o,d]

    for o_surrounding in o_surroundings: #先把O点相邻的路口添加到all_paths里 再以此为基础扩散
        all_paths.append([o,o_surrounding])

    delta_all_paths_len = 0

    #开始遍历搜索 默认搜索10层 太多了反而没意义 应该思考是不是数据有问题 竟然缺了10个路口的数据
    for _ in range(search_times):
        all_paths_len = len(all_paths) #记录一下当前的路径数
        new_paths_indexs = range(all_paths_len-delta_all_paths_len,all_paths_len) if delta_all_paths_len != 0 else range(all_paths_len)
        for i in new_paths_indexs: #遍历已有路径集中的新鲜路径
            path = all_paths[i]                    
            temp_d = path[-1] #获取路径中的d点
            #然后以这个临时的D点为主 向四周发散相邻格
            temp_d_surroundings = map_info[temp_d] #获取临时D点相邻的路口

            #如果临时D点相邻的路口中就包含我们在找的D点 那么直接返回就行
            if d in temp_d_surroundings:
                shortest_path = list(path) #复制一次该路径 等会儿要在这个路径后边加东西
                shortest_path.append(d)

                print("找到了{}-{}之间的最短路。".format(o,d))
                return shortest_path

            #如果临时D点相邻的路口中不包含我们在找的D点 那就继续找
            for temp_d_surrounding in temp_d_surroundings:
                new_path = list(path) #复制一次该路径 等会儿要在这个路径后边加东西
                if  temp_d_surrounding not in new_path: #如果temp_d_surrounding没有出现在路径里 那么就添加进去（避免无限循环回头路）
                    new_path.append(temp_d_surrounding)
                    #然后看看这个新路径有没有重复的 没有就添加到all_paths里
                    if new_path not in all_paths:
                        all_paths.append(new_path)

        delta_all_paths_len = len(all_paths) - all_paths_len #计算循环前后路径数量的差值 （因为每次其实只需要对新加进来的路径进行循环扩充，不需要对已有的路径进行操作）

    print("找不到{}-{}之间{}个路口内的最短路。".format(o,d,search_times))
    return shortest_path


#搜索给定OD之间的（指定数量上限的）线路（限定搜索次数 也就是限定路线长度） ###慎用！该方法使用穷举 计算时间很长
def OD_All_Paths(o,d,map_info,search_times,max_length):
    o_d_all_paths = [] #满足OD要求的路线

    all_paths = [] #从O点发散搜索到的所有路径
    o_surroundings = map_info[o] #获取O点相邻的路口
    for o_surrounding in o_surroundings: #先把O点相邻的路口添加到all_paths里 再以此为基础扩散
        all_paths.append([o,o_surrounding])
        if o_surrounding == d: #如果相邻路口中就有D点 那么先添加进去
            o_d_all_paths.append([o,o_surrounding]) #添加到输出列表里

    delta_all_paths_len = 0
    #开始遍历搜索 最多搜索search_times次
    for _ in range(search_times):
        all_paths_len = len(all_paths) #记录一下当前的路径数
        new_paths_indexs = range(all_paths_len-delta_all_paths_len,all_paths_len) if delta_all_paths_len != 0 else range(all_paths_len)
        for i in new_paths_indexs: #遍历已有路径集中的新鲜路径
            path = all_paths[i]
            temp_d = path[-1] #获取路径中的d点
            #然后以这个临时的D点为主 向四周发散相邻格
            temp_d_surroundings = map_info[temp_d] #获取临时D点相邻的路口
            for temp_d_surrounding in temp_d_surroundings:
                new_path = list(path) #复制一次该路径 等会儿要在这个路径后边加东西
                if  temp_d_surrounding not in new_path: #如果temp_d_surrounding没有出现在路径里 那么就添加进去（避免无限循环回头路）
                    new_path.append(temp_d_surrounding)
                    #然后看看这个新路径有没有重复的 没有就添加到all_paths里
                    if new_path not in all_paths:
                        all_paths.append(new_path)
                        #同时统计一下该路径是否符合OD
                        if new_path[-1] == d: #如果这条路径的终点就是我们要找的d点
                            o_d_all_paths.append(new_path) #添加到输出列表里
                            #判断是否已经搜索足够数量的路径了
                            if len(o_d_all_paths) >= max_length: #如果已经满足要求 那么就返回
                                return o_d_all_paths

        delta_all_paths_len = len(all_paths) - all_paths_len #计算循环前后路径数量的差值 （因为每次其实只需要对新加进来的路径进行循环扩充，不需要对已有的路径进行操作）

    # #all_paths全部生成完以后 寻找符合要求的路线
    # o_d_all_paths = [] #满足OD要求的路线
    # for path in all_paths:
    #     if path[-1] == d: #如果这条路径的终点就是我们要找的d点
    #         o_d_all_paths.append(path)

    return o_d_all_paths


#根据路网拓扑表字典构造任意两点之间的最短路表
def Generate_Shortest_Paths(map_info,search_times,o_start=1,o_end=370):
    shortest_paths = {}
    for o in range(o_start,o_end):
        for d in range(o_start,o_end):
            if o == d:
                continue
            if d > o: #如果终点序号大于起点序号 那就正常进行
                shortest_path = OD_Shortest_Paths(o,d,map_info,search_times)
                #找到最短路后 就直接记录下来吧
                shortest_paths[str(o) + "-" + str(d)] = shortest_path
            else: #如果终点序号小于起点序号 说明这个OD之间的最短路已经被找过了 只需要调换OD 添加到列表就行
                paths = shortest_paths[str(d) + "-" + str(o)]
                shortest_paths[str(o) + "-" + str(d)] = Reverse_OD_Paths(paths,paths_type="shortest")
    return shortest_paths

#根据路网拓扑表字典构造任意两点之间的可行路径表 （限制数量）
def Generate_All_Possible_Paths(map_info,search_times,max_length,o_start=1,o_end=370):
    all_possible_paths = {}
    for o in range(o_start,o_end):
        for d in range(o_start,o_end):
            if o == d:
                continue
            if d > o: #如果终点序号大于起点序号 那就正常进行
                all_possible_path = OD_All_Paths(o,d,map_info,search_times,max_length)
                #找到可行路径集后 就直接记录下来吧
                all_possible_paths[str(o) + "-" + str(d)] = all_possible_path
            else: #如果终点序号小于起点序号 说明这个OD之间的可行路径已经被找过了 只需要调换OD 将路段反转 添加到列表就行
                paths = all_possible_paths[str(d) + "-" + str(o)] #先获取
                all_possible_paths[str(o) + "-" + str(d)] = Reverse_OD_Paths(paths,paths_type="all")

            #输出一些log
            paths_count = len(all_possible_path)
            if paths_count > 0:
                print("找到了{}-{}之间的可行路径{}条。".format(o,d,paths_count))
            else:
                print("找不到{}-{}之间{}个路口内的可行路径。".format(o,d,search_times))

    return all_possible_paths

#小函数 反转两个OD之间的路径集 比如 1-2-3-4 变成 4-3-2-1
def Reverse_OD_Paths(paths:list,paths_type="all") -> list:
    reversed_paths = []
    if paths_type == "all":
        #说明传进来的paths含有很多子列表
        for path in paths:
            reversed_paths.append([ele for ele in reversed(path)])

    if paths_type == "shortest":
        #说明传进来的paths就是一个单独的列表 直接反转就行
        reversed_paths = [ele for ele in reversed(paths)]

    return reversed_paths

#最短路算法修补车辆路径
def Generate_Real_Paths(paths_origin_with_time,shortest_paths,missing_cross_id=[],max_len=20):
    #遍历原始数据中的车辆路径 （paths_origin_with_time是一个字典）
    final_paths = {} #修补后的路径字典
    final_paths_with_time ={} #修补后的路径字典(含经过各点的时间)
    wrong_datas = {} #记录修补有问题的数据
    veh_ids = list(paths_origin_with_time.keys()) #车辆的ID列表
    for veh_id in veh_ids:
        path = paths_origin_with_time[veh_id][0] #读取这辆车的原始出行路径
        path_time = paths_origin_with_time[veh_id][1] #读取这辆车原始出行路径中经过每个路口的时间
        wrong_data = [] #记录有问题的片段

        #针对这个出行路径，将其相邻路口一一拆分 分别使用最短路进行修补
        #注意：path_origin中是包含OD的 元素分别位于列表第一个和最后一个位置
        path_len = len(path) #获取路径长度

        if path_len < 2: #如果路径长度只有1 那么就是只检测到了这辆车的一次数据，直接保留就行（其实这一步在原始数据的拼接时已经考虑了，当时设置了min_len来过滤那种数据量少的车
            final_paths[veh_id] = path
            final_paths_with_time[veh_id] = [path,path_time]
            continue

        if path_len > max_len: #如果路径长度大于20 那么很可能这个车牌有问题(仅限一小时内的数据 如果全天的数据 是很有可能大于20的) 直接忽略这辆车的数据
            continue

        reCon_Path = [] #存放重构后的结果
        reCon_Path_time = [] #存放重构后的结果 时间
        #对路径内每两个相邻的路段ID进行修补
        for i in range(len(path)-1):
            cross_id_o = path[i]
            cross_id_d = path[i+1]
            cross_id_o_time = path_time[i]
            cross_id_d_time = path_time[i+1]
            p = shortest_paths.get(str(cross_id_o) + "-" + str(cross_id_d),[cross_id_o,cross_id_d]) #获取这两个路段之间的最短路（不管这两个路段是否相邻，因为相邻的话也会返回正确的结果）
            #如果找不到这两个路口之间的最短路 说明数据有点问题 车辆位置跳跃有点大 那就直接保留下来吧 这一段不进行修补 但是要记录下来
            if p == []: 
                reCon_Path.append(cross_id_o)
                reCon_Path.append(cross_id_d)
                reCon_Path_time.append(cross_id_o_time)
                reCon_Path_time.append(cross_id_d_time)
                wrong_data.append(str(cross_id_o) + "-" + str(cross_id_d))
            else:
                #循环写入reCon_Path中
                for item in p :
                    reCon_Path.append(item)
                    if item == p[0]:
                        reCon_Path_time.append(cross_id_o_time) #原样记录通过时刻
                    elif item == p[-1]:
                        reCon_Path_time.append(cross_id_d_time) #原样记录通过时刻
                    else:
                        reCon_Path_time.append(-1) #对于修补出来的路径 那就直接补-1作为通过时刻
            if i == len(path)-2 : #如果不是最后一个i的话 那就每次添加后把reCon_Path的最后一项删了 因为会重复
                break
            reCon_Path.pop()
            reCon_Path_time.pop()

        #记录有问题的部分 这些车辆直接删掉 不存到路径集中
        if len(wrong_data) > 0:
            wrong_datas[veh_id] = wrong_data
        else:
            #修补完成后直接存入字典中
            final_paths[veh_id] = reCon_Path
            final_paths_with_time[veh_id] = [reCon_Path,reCon_Path_time]


    print("最短路算法车辆路径已生成，共{}条。".format(len(final_paths)))
    print("有问题的车辆路径数为{}条。".format(len(wrong_datas)))
    #将wrong_datas存为文件
    np.save("outputs\\real_paths_wrong_datas.npy",wrong_datas)
    return final_paths,final_paths_with_time

#根据缺失指定路口的车辆路径集构造一个用来计算算法误差的车辆路径集（因为如果路口缺失过多，很可能使得一些车辆的数据直接消失，无法重构，这对误差计算影响很大）
def Gen_Real_Paths_For_Error_Calculation(Lost_Path_Datas,real_paths_By_Shortest_Method,min_path_len=1):
    real_paths_for_error_calculation = {}
    for veh_id in list(real_paths_By_Shortest_Method.keys()):
        lost_path = Lost_Path_Datas.get(veh_id,[])
        real_path = real_paths_By_Shortest_Method[veh_id]
        #判断是否满足长度要求
        if len(lost_path) >= min_path_len:
            real_paths_for_error_calculation[veh_id] = real_path

    return real_paths_for_error_calculation



#将paths字典储存为TXT文件
def Dict_to_CSV(paths_dict,filename,mode="without_time"):
    if mode == "without_time": #传入的paths_dict不含时间数据
        paths_ls = []
        veh_ids = list(paths_dict.keys()) #车辆的ID列表
        for veh_id in veh_ids:
            path = paths_dict[veh_id] #读取这辆车的出行路径
            paths_ls.append([veh_id,path])

        with open(filename, 'w',encoding="utf-8") as f:
            f.write("车牌:车辆出行路径\n")
            for item in paths_ls:
                f.write(item[0][0:32]+":")
                s = ""
                for x in item[1]:
                    s += str(x) + ","
                f.write(s)
                f.write('\n')

    elif mode == "with_time": #传入的paths_dict包含时间数据
        paths_ls = []
        veh_ids = list(paths_dict.keys()) #车辆的ID列表
        for veh_id in veh_ids:
            path = paths_dict[veh_id][0] #读取这辆车的出行路径
            path_time = paths_dict[veh_id][1] #读取这辆车的出行时间
            paths_ls.append([veh_id,path,path_time])

        with open(filename, 'w',encoding="utf-8") as f:
            f.write("车牌:车辆出行路径\n")
            for item in paths_ls:
                f.write(item[0][0:32]+":")
                s = ""
                for i in range(len(item[1])):
                    p = item[1][i]
                    t = item[2][i]
                    s += str(p) + "," + str(t) + "," 
                f.write(s)
                f.write('\n')

#将路段流量输出成CSV
def Link_Flow_to_CSV(link_flow:dict,filename):
    temp_ls = []
    for key in list(link_flow.keys()):
        key_str = ""
        for i in range(len(key)):
            item = key[i]
            key_str += str(item) + "->" if i < len(key) - 1 else str(item)
        temp_ls.append([key_str,link_flow[key]])
    
    #使用pandas导出结果
    temp_pd = pd.DataFrame(columns=["Link","Flow"],data=temp_ls)
    temp_pd.to_csv(filename,index=False)


#将路径流量输出成CSV
def Path_Flow_to_CSV(path_flow:dict,filename):
    temp_ls = []
    for key in list(path_flow.keys()):
        key_str = ""
        for i in range(len(key)):
            item = key[i]
            key_str += str(item) + "->" if i < len(key) - 1 else str(item)
        temp_ls.append([key_str,path_flow[key]])
    
    #使用pandas导出结果
    temp_pd = pd.DataFrame(columns=["Path","Flow"],data=temp_ls)
    temp_pd.to_csv(filename,index=False)

#将路径流量输出成CSV
def OD_Distribution_to_CSV(OD_distribution:dict,filename):
    temp_ls = []
    for key in list(OD_distribution.keys()):
        temp_ls.append([key,OD_distribution[key]])
    
    #使用pandas导出结果
    temp_pd = pd.DataFrame(columns=["OD_Distribution_Index","Flow"],data=temp_ls)
    temp_pd.to_csv(filename,index=False)

#根据路径集path和相邻路口信息links_info统计一下每个路段的流量 同时记录一下出错的数据（路径中两个相邻路口之间距离过远） !!!统计有限制的，认为只有路径长度大于等于min_path_len的才能统计
def Calc_Link_Flow(paths: dict,links_info:dict,min_path_len=0):
    veh_ids_ls = list(paths.keys()) #所有车辆ID
    path_ls = list(paths.values()) #所有路径
    
    link_flow = {} #路段流量统计表
    wrong_datas = [] #记录有问题的数据

    link_od_ls = list(links_info.keys()) #获取所有路段OD

    #初始化统计表
    for link_od in link_od_ls:
            link_flow[link_od] = 0


    #遍历每辆车的路径
    for i in range(len(path_ls)):
        path = path_ls[i] #找到这辆车的路径集
        #判断这辆车的路径是否为有效路径(是否满足最短路径长度的要求)
        if len(path) < min_path_len:
            continue
        #构造路口-路口之间的路段ID
        for j in range(len(path)-1):
            link_od = (path[j],path[j+1])
            #判断是否存在这个路段（如果两个路口之间距离太远的话是不存在直达路段的）
            if link_od in link_od_ls:
                link_flow[link_od] = link_flow[link_od] + 1
            else:
                wrong_datas.append([veh_ids_ls[i],link_od])

    return link_flow, wrong_datas

#计算两种重构结果的路段流量差 a - b
def Calc_Delta_Link_Flow(recon_link_flow,real_link_flow):
    delta_link_flow = {}
    for link_id in list(recon_link_flow.keys()):
        recon = recon_link_flow[link_id]
        real = real_link_flow[link_id]
        delta_link_flow[link_id] = recon - real
    
    return delta_link_flow


#计算车辆路径集paths的路径流量字典
def Calc_Path_Flow(paths:dict) -> dict:
    path_flow = {}
    for path in list(paths.values()):
        path_tuple = tuple(path) #作为字典的key
        path_flow[path_tuple] = path_flow.get(path_tuple,0) + 1
    
    return path_flow


#根据路径长度获取长度分布所属组别
def Get_OD_Distribution_Index(distance,group_count,step=1.0):
    index = group_count-1 #默认组别是最后一组
    for i in range(0,group_count):
        if distance <= (i+1)*step: #如果距离小于这个组别的右界，那么它就是这个组别的
            return i

    return index

#统计路径重构结果的OD距离及分布
def Calc_OD_distribution(path_flow,links_info,group_count=40,step=1.0): #按长度为1km来分组
    real_OD_distribution = {}
    #先初始化这个统计字典 分成50组 每组间距为step
    for i in range(group_count):
        real_OD_distribution[i] = 0

    #然后开始遍历整个path_flow 计算每个路径的长度
    path_length = {}
    for path in list(path_flow.keys()):
        distance = 0.0
        for i in range(len(path)-1): #两两计算距离
            od = (path[i],path[i+1])
            info = links_info.get(od,{"长度":200}) #默认长度给个200吧
            length = info["长度"]
            distance += length/1000 #单位 km

        path_length[path] = distance #把距离记录下来

    
    #然后开始遍历分组计数
    for path in list(path_flow.keys()):
        distance = path_length[path] #获取长度
        flow = path_flow[path] #获取流量
        index = Get_OD_Distribution_Index(distance,group_count,step=step) #获取所属组别
        real_OD_distribution[index] += flow #对应组别流量计数累计
    
    return real_OD_distribution


#根据路径集path和相邻路口信息map_info统计一下每个路口的流量
def Calc_Current_Nodes_volumn(paths: dict,cross_ids:dict):
    nodes_volumn = {}

    path_ls = list(paths.values()) #所有路径
    
    #初始化统计表
    for cross_id in cross_ids:
        nodes_volumn[cross_id] = 0

    #遍历车辆路径来统计
    for path in path_ls:
        for node in path:
            nodes_volumn[node] += 1

    return nodes_volumn


#根据给出的放大倍数 来生成路口流量最大限制字典
def Calc_Max_Nodes_volumn(node_volumn,scale=1.25):
    max_node_volumn = {}
    nodes_ls = list(node_volumn.keys())
    for node in nodes_ls:
        max_node_volumn[node] = node_volumn[node] * scale if node_volumn[node] > 0 else (node_volumn[node] + 500) * scale #有些路口可能是没有数据的 那么给他一个基础值 500
    
    return max_node_volumn


#车辆路径数据简单分析
def Paths_Datas_Analysis(ax,recon_paths:dict,real_paths:dict,len_lim=11):
    #数据分析与统计

    #1.车辆数（路径数）
    recon_paths_count = len(recon_paths)
    real_paths_count = len(real_paths)

    #2.路径长度分布图 （1,2,3,4,5,6,7,8,9,10...）
    recon_paths_length_distribution = {}
    real_paths_length_distribution = {}
    for i in range(1,len_lim):
        recon_paths_length_distribution[i] = 0
        real_paths_length_distribution[i] = 0

    recon_paths_ls = list(recon_paths.values())
    real_paths_ls = list(real_paths.values())
    for path in recon_paths_ls:
        path_length = len(path)
        if path_length < len_lim:
            recon_paths_length_distribution[path_length] += 1

    for path in real_paths_ls:
        path_length = len(path)
        if path_length < len_lim:
            real_paths_length_distribution[path_length] += 1
    
    print(f"重构前路径数量：{real_paths_count} 条，重构后路径数量：{recon_paths_count} 条。","\n")

    #画柱状图
    # ax = plt.subplot(122)
    x = np.arange(1,len_lim)
    a = list(recon_paths_length_distribution.values())
    b = list(real_paths_length_distribution.values())
    width = 0.4
    ax.bar(x - width/2, a,  width=width, label='recon paths')
    ax.bar(x + width/2, b, width=width, label='real paths')
    if len_lim < 21: #如果柱数小于21 那就显示X轴标签
        ax.set_xticks(np.arange(1,len_lim))
    if len_lim < 12: #如果柱数小于12 那就显示数据标签
        for i,j in zip(x,a):
            plt.text(i - width/2, j+0.05, '%.0f' % j, ha='center', va= 'bottom',fontsize=7)
        for i,j in zip(x,b):
            plt.text(i + width/2, j+0.05, '%.0f' % j, ha='center', va= 'bottom',fontsize=7)
    ax.set_xlabel('Path Length')
    ax.set_ylabel('Path Count')
    ax.legend()

    ax.set_title("Paths Length Distribution")


#计算 MAE RMSE R2
def Calc_MAE_RMSE_R2(re_ls,real_ls):
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    mae = mean_absolute_error(real_ls, re_ls)
    # mse = mean_squared_error(real_ls, re_ls)
    rmse = math.sqrt(mean_squared_error(real_ls, re_ls))
    r2 = r2_score(real_ls, re_ls) if len(real_ls)>1 and len(re_ls)>1 else 1.0
    return mae,rmse,r2

#评估函数 评估两组路径的路段流量差异
def Evaluate_Links(ax, real_path_link_flow:dict, re_path_link_flow:dict,lim=3000):
    #1.统计各个路段的流量 画图  横轴真实流量 纵轴重构流量 
    re_path_link_flow_ls = list(re_path_link_flow.values())
    real_path_link_flow_ls = list(real_path_link_flow.values())
    # ax = plt.subplot(121)
    ax.set_xlim(xmax=lim,xmin=0) #横轴刻度
    ax.set_ylim(ymax=lim,ymin=0) #纵轴刻度
    ax.set_xlabel('Real Link Flow (veh)')
    ax.set_ylabel('Reconstruction Link Flow (veh)')
    ax.scatter(real_path_link_flow_ls, re_path_link_flow_ls,s=5,c="red",zorder=10)
    ax.plot([0,lim], [0,lim], linewidth=1,zorder=5)
    ax.set_title("Links Flow")

    
    #计算 MAE RMSE R2
    mae,rmse,r2 = Calc_MAE_RMSE_R2(real_path_link_flow_ls,re_path_link_flow_ls)
    
    #标注mae,rmse,r2
    # ax.text(lim*0.1, lim*0.9, f"MAE: {mae:0.1f}, RMSE: {rmse:0.1f}, R2: {r2:0.2f}", fontsize=10) 
    ax.text(lim*0.1, lim*0.9, f"MAE: {mae:0.1f}, RMSE: {rmse:0.1f}", fontsize=10) 
    # print(f"MAE: {mae:0.1f}, RMSE: {rmse:0.1f}, R2: {r2:0.2f}")
    return mae,rmse,r2


#评估函数 评估两组路径的路径流量差异
def Evaluate_Paths(ax,re_path_path_flow:dict, real_path_path_flow:dict, lim=3000, min_path_flow_limit=0,title="Path Flow"):
    #构造两个对齐的列表 这样才能对比

    #先合并两个字典的keys 构建一个值全为0的列表
    all_keys = re_path_path_flow.copy() #复制一次
    all_keys.update(real_path_path_flow) #合并字典
    #然后只取字典的keys 构造新字典
    recon = {}
    real = {}
    for key in list(all_keys.keys()):
        recon[key] = 0
        real[key] = 0
    
    #初始化完毕后 各自更新一下
    for key in list(re_path_path_flow.keys()):
        recon[key] = re_path_path_flow[key]
    
    for key in list(real_path_path_flow.keys()):
        real[key] = real_path_path_flow[key]

    #然后根据limit处理一下
    final_recon = {}
    final_real = {}
    for key in list(real.keys()): #遍历所有路径
        if real[key] >= min_path_flow_limit: #如果真实路径集中，这条路径的流量大于等于limit，那就保留
            final_recon[key] = recon[key]
            final_real[key] = real[key]

  
    #1.统计各个路段的流量 画图  横轴真实流量 纵轴重构流量 
    re_path_path_flow_ls = list(final_recon.values())
    real_path_path_flow_ls = list(final_real.values())
    # ax = plt.subplot(121)
    ax.set_xlim(xmax=lim,xmin=0) #横轴刻度
    ax.set_ylim(ymax=lim,ymin=0) #纵轴刻度
    ax.set_xlabel('Real Path Flow (veh)')
    ax.set_ylabel('Reconstruction Path Flow (veh)')
    ax.scatter(real_path_path_flow_ls, re_path_path_flow_ls,s=5,c="red",zorder=10)
    ax.plot([0,lim], [0,lim], linewidth=1,zorder=5)
    ax.set_title(title)
    #计算 MAE RMSE R2
    mae,rmse,r2 = Calc_MAE_RMSE_R2(re_path_path_flow_ls,real_path_path_flow_ls)
    #标注mae,rmse,r2
    ax.text(lim*0.1, lim*0.9, f"MAE: {mae:0.1f}, RMSE: {rmse:0.1f}, R2: {r2:0.2f}", fontsize=10) 
    return #mae,rmse,r2


#计算路径流量误差： 只计算流量大于limit的路径
def Calc_Path_Flow_Error(recon_path_flow,real_path_flow,min_path_flow_limit = 10):
    #构造两个对齐的列表 这样才能对比

    #先合并两个字典的keys 构建一个值全为0的列表
    all_keys = recon_path_flow.copy() #复制一次
    all_keys.update(real_path_flow) #合并字典
    #然后只取字典的keys 构造新字典
    recon = {}
    real = {}
    for key in list(all_keys.keys()):
        recon[key] = 0
        real[key] = 0
    
    #初始化完毕后 各自更新一下
    for key in list(recon_path_flow.keys()):
        recon[key] = recon_path_flow[key]
    
    for key in list(real_path_flow.keys()):
        real[key] = real_path_flow[key]

    #然后根据limit处理一下
    final_recon = {}
    final_real = {}
    for key in list(real.keys()): #遍历所有路径
        if real[key] >= min_path_flow_limit: #如果真实路径集中，这条路径的流量大于等于limit，那就保留
            final_recon[key] = recon[key]
            final_real[key] = real[key]

    #现在可以计算误差了
    #计算 MAE RMSE R2
    mae,rmse,r2 = Calc_MAE_RMSE_R2(list(final_recon.values()),list(final_real.values()))

    return mae,rmse,r2

#根据指定丢失的路口编号 生成对应的车辆路径（不完整） 以模拟低渗透率的情况
def Gen_Lost_Path_Datas(lost_cross_ID:list,paths_By_Shortest_Method_with_time:dict,min_len=2):
    min_len = 1 if min_len < 1 else min_len
    Lost_Paths = {}
    Lost_Paths_with_time = {}

    veh_ids = list(paths_By_Shortest_Method_with_time.keys()) #获取车辆ID
    paths = list(paths_By_Shortest_Method_with_time.values()) #获取路径集
    for i in range(len(veh_ids)):
        path = paths[i][0] #获取这辆车的路径
        path_time = paths[i][1] #获取这辆车的时间
        new_path = []
        new_path_time = []
        for j in range(len(path)):
            cross = path[j]
            time = path_time[j]
            if cross not in lost_cross_ID:
                if len(new_path) > 0:
                    if cross != new_path[-1]: #避免添加两个连续的相同路口 后续程序会出错 
                        new_path.append(cross)
                        new_path_time.append(time)
                else:
                    new_path.append(cross)
                    new_path_time.append(time)  

        #对路径长度有限制，小于2个数据点的路径就不要添加了
        if len(new_path) >= min_len:
            Lost_Paths[veh_ids[i]] = new_path
            Lost_Paths_with_time[veh_ids[i]] = [new_path,new_path_time]

    print("模拟数据缺失情况的车辆路径已生成，共{}条，路口缺失率{:.1%}。模拟缺失的路口ID：".format(len(Lost_Paths),len(lost_cross_ID)/114),lost_cross_ID,"\n")

    return Lost_Paths,Lost_Paths_with_time


#给定一个数据缺失率 生成符合缺失率的列表
def Gen_Lost_CrossIDs(lost_rate, cross_id_start = 1, cross_id_end = 114) ->list:
    lost_cross_IDs = []
    lost_count = math.floor(cross_id_end * lost_rate) #计算需要选取多少个路口 （向下取整）

    # for i in range(cross_id_start,cross_id_end+1):
    #     r = random.random()
    #     if r < lost_rate:
    #         lost_cross_IDs.append(i)

    lost_cross_IDs = random.sample(list(range(cross_id_start,cross_id_end+1)),lost_count)
    lost_cross_IDs.sort() #排序一下 方便人看

    return lost_cross_IDs


#给定一对OD给出连接它们的可能路径集合 如果找不到 那就给最短路的 #不要给太多了 只要5个
def Gen_OD_Possible_Paths(o,d,all_possible_paths,shortest_paths,max=5):
    # if str(o) == str(d):
    #     r = [[o,d]]
    #     return r

    r = all_possible_paths[str(o) + "-" + str(d)]
    if len(r) == 0:
        s = shortest_paths[str(o) + "-" + str(d)]
        r = [s]
    return r[:max]

#两两连接
def Combine(a,b):
    r = []
    for i in a:
        for j in b:
            x = i+j #列表连接
            r.append(x)
    return r

#数学问题 给出一个列表，生成这个列表的单向连接组合
def Gen_Repair_Solution_Indexs(flag):
    t = []
    for item in flag:
        temp = []
        for x in list(range(item)):
            temp.append([x])
        t.append(temp)

    a = t[0]
    for i in range(1,len(t)):
        b = t[i]
        a = Combine(a,b)

    return a

#输入车辆的残缺路径，使用all_possible_paths组合出车辆的可能路径集 (更新：lost_cross_IDs也要用于构造车辆OD点的修补)
def Gen_PF_Possible_Paths(path_lost,path_time,lost_cross_IDs,all_possible_paths,shortest_paths,map_info,paths_count_lim=100):
    paths_possible = [] #可行路径集
    paths_w = [] #路径权重
    paths_possible_time = [] #可行路径集中经过每个路口的时刻
    #遍历路径中的每个相邻路口
    path_len = len(path_lost) #获取路径长度

    if path_len < 2: #如果路径长度只有1 那么就是只检测到了这辆车的一次数据，直接保留就行
        paths_possible = list(path_lost) #复制一次
        paths_possible_time = list(path_time) #复制一次
        return [paths_possible],[1.],[paths_possible_time]
    else: 
        #对路径内每两个相邻的路段ID进行修补
        #1.先分析一下路径 看看需要修补几次 分别在哪个地方修补
        i_need_repair = [] #记录需要修补的o点下标
        o_d_need_repair = [] #记录需要修补的OD对
        for i in range(path_len-1):
            cross_id_o = path_lost[i]
            cross_id_d = path_lost[i+1]
            #找一下D点是不是在O点相邻的路口
            #如果D点不在O点相邻的路口中，那么记录下来 说明这一段需要修补
            if cross_id_d not in map_info[cross_id_o]:
                i_need_repair.append(i)
                o_d_need_repair.append([cross_id_o,cross_id_d])
        
        #判断一下这条路径有没有需要修补的地方 没有的话直接返回就行
        if len(i_need_repair) == 0:
            paths_possible = list(path_lost) #复制一次
            paths_possible_time = list(path_time) #复制一次
            return [paths_possible],[1.],[paths_possible_time]

        #分析完以后 如果需要修补 开始针对缺失的部分逐个去找材料修补
        repair_store = [] #修补的仓库
        for lost_o_d in o_d_need_repair:
            repair_store.append(Gen_OD_Possible_Paths(lost_o_d[0],lost_o_d[1],all_possible_paths,shortest_paths,max=3))

        #倒序一下 为了方便后续修补 不改变前边其他待修补部分的index
        i_need_repair.reverse() # = list(reversed(i_need_repair))
        repair_store.reverse() # = list(reversed(repair_store))

        #找到了各个部分的修补材料后 开始逐个拼接起来生成可能的路径集
        flag = [] #将问题转化成下标的组合问题 另外写一个函数去生成组合 同时计算一下有多少种可行路径组合
        possible_paths_count = 1
        for item in repair_store:
            flag.append(len(item))
            possible_paths_count = possible_paths_count * len(item)
        
        #如果可行路径的数量太多了 那就只取最短路连接，不然内存不够（这些车也是少数，影响不大）
        if possible_paths_count > 8192:
            flag = [1 for _ in flag]
        
        #求解可行方案的下标组合
        repair_solution_indexs = Gen_Repair_Solution_Indexs(flag)
        #限制可行路径的数量
        repair_solution_indexs = repair_solution_indexs[0:paths_count_lim]

        #拿到解后，直接根据解来构造完整的可行路径
        for solution in repair_solution_indexs:
            #先复制一份原路径
            new_path = list(path_lost)
            #复制一份原时间
            new_path_time = list(path_time)
            #然后倒序修补 从最后往前 这样可保证前边的index不乱（之前已经翻转过了）
            for i in range(len(solution)):
                s = solution[i] #获取仓库内容的下标
                #找到仓库中对应的修补内容
                part = repair_store[i][s]
                #把part掐头去尾 因为part是包含O点和D点的
                part = part[1:-1]
                #倒序part
                part = list(reversed(part))
                #将part中的内容一一插入到原始路径中需要修补的位置
                for p in part:
                    new_path.insert(i_need_repair[i]+1,p)
                    new_path_time.insert(i_need_repair[i]+1,-1) #重构路径的时间用-1来补位
            #修补完成后添加到输出列表中
            paths_possible.append(new_path)
            paths_possible_time.append(new_path_time)


##############↓↓↓↓↓已弃用↓↓↓↓↓###################
    #中间的部分全部修补完成以后，另外处理O和D两点：OD点可能不是真的OD点，可以尝试从OD点周围找相邻路口加进去作为真的OD点（但是也有问题，到底加几个？）
    o = path_lost[0]
    d = path_lost[-1]
    #找到OD点相邻的所有缺失的路口
    o_missing_surroundings = [item for item in map_info[o] if item in lost_cross_IDs]
    d_missing_surroundings = [item for item in map_info[d] if item in lost_cross_IDs]

    #2021.06.10修改：先不加OD两个点周围的路口  以后再试试多加更多的OD延伸点：从shortest_paths里找这个点不为空的最短路，加进去看看  其实主要问题是怎么才能生成合理的初始路径集
    o_missing_surroundings = []
    d_missing_surroundings = []

    #先处理O点 在O点之前加上所有可能的真实O点
    temp_paths_possible = []
    for o_possible in o_missing_surroundings:
        for p in paths_possible:
            temp_p = list(p) #复制一次path
            temp_p.insert(0,o_possible) #在path开头插入o_possible
            temp_paths_possible.append(temp_p) #将插入后的路径添加到临时列表里

    #处理完后更新一下paths_possible 如果o_missing_surroundings不是空的 那么才更新
    paths_possible = temp_paths_possible if len(o_missing_surroundings) > 0 else paths_possible

    #再处理D点 在D点之后加上所有可能的真实D点
    temp_paths_possible = []
    for d_possible in d_missing_surroundings:
        for p in paths_possible:
            temp_p = list(p)
            temp_p.append(d_possible)
            temp_paths_possible.append(temp_p)

    #处理完后更新一下paths_possible
    paths_possible = temp_paths_possible if len(d_missing_surroundings) > 0 else paths_possible

##############↑↑↑↑↑已弃用↑↑↑↑↑###################

    #处理一下权重表 归一化权重
    l = len(paths_possible) 
    for _ in paths_possible:
        paths_w.append(0.99/l)

    #随机给一个路径较大的初始权重值
    r = random.randrange(0,len(paths_w))
    paths_w[r] = paths_w[r] + 0.01 

    # paths_w[0] = paths_w[0] + 0.01 #给第一个路径较大的初始权重值

    return paths_possible, paths_w, paths_possible_time

#输出每辆车权重最大的那个路径
def Calc_Current_Paths(veh_datas):
    current_paths = {}
    for veh_id in list(veh_datas.keys()):
        paths_possible = veh_datas[veh_id][0]
        paths_w = veh_datas[veh_id][1]
        max_w = -1.
        max_w_i = -1
        for i in range(len(paths_w)):
            w = paths_w[i]
            if w > max_w:
                max_w = w
                max_w_i = i

        current_paths[veh_id] = paths_possible[max_w_i]

    return current_paths

#输出每辆车权重最大的那个路径和经过各个路口的时间
def Calc_Current_Paths_with_time(veh_datas):
    current_paths_with_time = {}
    for veh_id in list(veh_datas.keys()):
        paths_possible = veh_datas[veh_id][0]
        paths_w = veh_datas[veh_id][1]
        paths_possible_time = veh_datas[veh_id][3]
        max_w = -1.
        max_w_i = -1
        for i in range(len(paths_w)):
            w = paths_w[i]
            if w > max_w:
                max_w = w
                max_w_i = i

        current_paths_with_time[veh_id] = [paths_possible[max_w_i],paths_possible_time[max_w_i]]

    return current_paths_with_time

#计算各个路段的流量大小和行程时间
def Calc_Links_Flow_and_Cost(paths,links_info,links_capacity):
    path_ls = list(paths.values()) #所有路径
    links_OD = list(links_info.keys()) #所有路口OD
    links_flow = {} #路段流量统计表
    links_cost = {} #路段行程时间统计表


    #初始化统计表
    for od in links_OD:
        # od_str = str(od[0])+"-"+str(od[1])
        # links_flow[od_str] = 0
        links_flow[od] = 0
        distance = links_info[od]["长度"] #单位：米
        speed_limit = links_info[od]["限速"]/3.6 #限速 单位 m/s
        # links_cost[od_str] = distance/speed_limit  #默认的通行时间是路段长度除以限速 单位：秒
        links_cost[od] = distance/speed_limit  #默认的通行时间是路段长度除以限速 单位：秒


    #遍历每辆车的路径
    for i in range(len(path_ls)):
        path = path_ls[i] #找到这辆车的路径集
        #构造路口-路口之间的路段ID
        for j in range(len(path)-1):
            # link_id = str(path[j])+"-"+str(path[j+1])
            link_id = (path[j],path[j+1])
            #判断是否存在这个路段（如果两个路口之间距离太远的话是不存在直达路段的）
            if link_id in links_OD:
                links_flow[link_id] = links_flow[link_id] + 1


    #路段的流量统计完毕以后 计算各个路段的行程时间 
    # linkcost=traveltime*(1+0.15*(volumn/capacity)^4)
    for link_id in links_OD:
        link_flow = links_flow[link_id] #获取流量
        # link_flow = links_flow.get(link_id,0) #获取流量
        capacity = max(links_capacity[link_id],600) #获取通行能力 最小值为600
        #粗略计算一下行程时间 美国经典路阻函数
        links_cost[link_id] = links_cost[link_id] * (1 + 0.15 * (link_flow/capacity)**4) 

    return links_flow, links_cost


#计算给定路段的出行费用 输入：自由流时间、路段当前流量，路段通行能力
def Calc_Link_Cost(link_free_cost,link_capacity,link_flow,a=0.15,b=4):
    cost = link_free_cost * (1 + a * (link_flow/link_capacity)**4) 
    return cost 




#给定一个path,根据links_cost来计算全程走完需要多少时间（秒）
def Calc_Path_Cost(path,links_cost):
    cost = 0.
    for i in range(len(path)-1):
        cross_id_o = path[i]
        cross_id_d = path[i+1]
        cost += links_cost.get(str(cross_id_o) + "-" + str(cross_id_d),9999)
    return cost

#给定一个path,根据links_cost来计算全程走完需要多少时间（秒）
def Calc_Path_Cost_PFE(path,links_datas):
    cost = 0
    for i in range(len(path)-1):
        cross_id_o = path[i]
        cross_id_d = path[i+1]
        # cost += links_datas.get(str(cross_id_o) + "-" + str(cross_id_d),9999)
        cost += links_datas[(cross_id_o,cross_id_d)][5]
    return cost


#行程时间一致性的函数 给出车辆路径的行驶时间和车辆真实的行驶时间 返回该条路径的概率 0<a<1<b<2 a、b是平均行程时间的上下限
def Calc_W_By_Cost(path_cost,real_cost,a,b):
    w = 0.

    if path_cost <= a*real_cost:
        w = 0.01
    elif a*real_cost < path_cost and path_cost <= real_cost:
        w = 2 - (real_cost/path_cost)
    elif real_cost < path_cost and path_cost <= b*real_cost:
        w = real_cost / path_cost
    elif b*real_cost <= path_cost:
        w = 0.01        

    return w

#找出每条路径中流量最小的路段 流量
def Calc_Min_Flow_In_Each_Path(links_flow,paths_possible):
    min_flow_in_each_path = []
    for path in paths_possible:
        min_flow = 999999
        for i in range(len(path)-1):
            # flow = links_flow[str(path[i])+"-"+str(path[i+1])]
            flow = links_flow.get((path[i],path[i+1]),0)
            if flow < min_flow:
                min_flow = flow
        min_flow_in_each_path.append(min_flow)

    return min_flow_in_each_path

#路径吸引力：路径流量大的路段，车辆选择的概率越大
def Calc_W_By_Attraction(i,min_flow_in_each_path):
    w = 1.
    sum_min_flow = sum(min_flow_in_each_path)

    w = min_flow_in_each_path[i] / sum_min_flow if sum_min_flow > 0 else 0.01

    return w


#OD修补路口流量约束 当流量比例达到limit时 概率为0 线形递减
def Calc_W_By_Node_Volumn(path,node_volumn,max_node_volumn,limit=1.25):
    w = 1.
    wo = 1.
    wd = 1.
    #判断OD点是不是经过修补的 如果不是 那就可以直接返回1
    #先获取OD点
    o = path[0]
    d = path[-1]
    #获取OD路口的流量
    o_volumn = node_volumn[o]
    d_volumn = node_volumn[d]
    #获取路口流量上限
    o_volumn_max = max_node_volumn[o]
    d_volumn_max = max_node_volumn[d]

    #如果流量比小于等于1 那么概率为1
    if o_volumn/o_volumn_max <=1 :
        wo = 1.
    else:
        if o_volumn/o_volumn_max < limit:
            wo = ( 0.9 /(limit - 1)) * (o_volumn/o_volumn_max - 1)

        else:
            wo = 0.1

    #如果流量比小于等于1 那么概率为1
    if d_volumn/d_volumn_max <=1 :
        wd = 1.
    else:
        if d_volumn/d_volumn_max < limit:
            wd = ( 0.9 /(limit - 1)) * (d_volumn/d_volumn_max - 1)

        else:
            wd = 0.1
    
    #最后综合两点的概率
    w = wo * wd

    return w


#读取进口方向——路段名称映射表 输出格式：{(路口ID,进口方向):对应的路段ID}
def Load_Direction_Map(filename):
    Direction_Map = {}

    #读取excel文件
    map_pd = pd.read_csv(filename, index_col = False)

    for _,row in map_pd.iterrows():
        O = row["路口ID"]
        Direction = row["进口方向"]
        From = row["来自路口ID"]
        Direction_Map[(O,Direction)] = (From,O)

    return Direction_Map

#从原始数据中计算每个布设有AVI的路段的流量观测值(要考虑那些实际上没有观测数据的点)
def Calc_AVI_Links_Flow(datas:DataFrame,Direction_Map:dict,lost_cross_ids:list):
    links_flow_avi = {}
    links_flow_avi_all = {} #临时的字典
    #初始化统计字典
    for value in Direction_Map.values():
        # 不统计那些数据缺失的路口
        if value[1] in lost_cross_ids:
            continue
        links_flow_avi[value] = 0

    #然后遍历原始AVI表 统计流量
    for _,row in datas.iterrows():
        D = row["路口ID"]
        Direction = row["进口方向"]
        links_flow_avi_all[(D,Direction)] = links_flow_avi_all.get(((D,Direction)),0) +1
        
    #然后再将路口ID-进口方向的流量统计字典转换成路段流量统计字典
    for key in Direction_Map.keys():
        # 不统计那些数据缺失的路口
        if key[0] in lost_cross_ids:
            continue

        link_id = Direction_Map[key]
        flow = links_flow_avi_all[key]
        
        links_flow_avi[link_id] = flow

    return links_flow_avi


#粒子滤波部分的权重更新迭代 输入上一轮的车辆数据和各路段行程时间 更新出新的车辆数据（就是计算新的权重）
def PF_Iteration(veh_datas,links_flow,links_cost,node_volumn,max_node_volumn):
    new_veh_datas = {}
    #遍历每一辆车的数据
    for veh_id in list(veh_datas.keys()):
        paths_possible = veh_datas[veh_id][0]
        travel_time = veh_datas[veh_id][2]
        paths_possible_time = veh_datas[veh_id][3]
        paths_w = [] #权重重新计算? 不使用旧的权重

        #2中需要的参数准备：
        #找出每条路径中流量最小的路段 流量
        min_flow_in_each_path = Calc_Min_Flow_In_Each_Path(links_flow,paths_possible)
        #更新每条可行路径的权重w
        for i in range(len(paths_possible)):
            path = paths_possible[i]
            w = 0.
            
            #1.行程时间一致性: 通过路径OD间的时间不能与平均时间偏差过大
            path_cost = Calc_Path_Cost(path,links_cost) #该条路径的行程时间
            real_cost = travel_time #车辆实际完成OD的时间
            a = Calc_W_By_Cost(path_cost,real_cost,a=0.95,b=1.05)



            #2.路径吸引力：路径流量大的路段，车辆选择的概率越大
            b = Calc_W_By_Attraction(i,min_flow_in_each_path)
            b = 1. #这个方程的效果较差，后期再调整



            #3.路径一致性(这里不需要了，因为在生成可行路径集的时候就把那些不合理的路径排除了，默认错检率为0%)
            c = 1


            #4.路口流量约束：OD两点的路口流量不能太离谱（主要是不能太大） 流量越小概率越高  越大概率越低 
            d = Calc_W_By_Node_Volumn(path,node_volumn,max_node_volumn,limit=1.25)


            #计算总权重
            w = a * b * c * d
            paths_w.append(w)

        new_veh_datas[veh_id]=[paths_possible,paths_w,travel_time,paths_possible_time]
    return new_veh_datas


#根据原始数据统计各个OD对之间的需求
def Count_OD_Flow(paths_origin:dict):
    OD_Flow = {}
    
    for path in list(paths_origin.values()):
        O = path[0]
        D = path[-1]
        if O == D: #不统计O与D相同的点（大概占2%~4%）
            continue
        OD = (O,D)
        OD_Flow[OD] = OD_Flow.get(OD,0) + 1

    return OD_Flow


#路径重构算法 使用缺失的数据集重构车辆路径(含时间数据)  PF粒子滤波方法
def Generate_PF_Paths(Lost_Path_Datas_with_time:dict,lost_cross_IDs:list,shortest_paths:dict,map_info:dict,cross_ids:dict,links_info:dict,links_capacity:dict,max_node_volumn:dict,iteration_count=10):
    re_paths = {} #存储最终结果
    #读取任意两个OD之间的可能路径集 后边要用到
    all_possible_paths = np.load('inputs\\all_possible_paths_10_5.npy',allow_pickle=True).item() #预制的任意两OD之间的可行路径集



    veh_datas = {}
    print("开始进行路径重构：\n","生成初始的可行路径集和权重...")
    #首先读取每辆车的路径和时间 生成初始的可行路径集和权重
    veh_ids = list(Lost_Path_Datas_with_time.keys())
    veh_ids_count = len(veh_ids)
    for i in range(len(veh_ids)):
        veh_id = veh_ids[i]
        path_lost = Lost_Path_Datas_with_time[veh_id][0] #读取残缺的车辆路径
        path_time = Lost_Path_Datas_with_time[veh_id][1] #读取车辆通过每个路口的时刻
        travel_time = path_time[-1] - path_time[0] #计算O-D的行程时间
        #生成初始的可行路径集和权重
        print("\r","正在处理第 {}/{} 辆车，veh_id:{}".format(i+1,veh_ids_count,veh_id), end=' ', flush=True)
        paths_possible, paths_w, paths_possible_time = Gen_PF_Possible_Paths(path_lost,path_time,lost_cross_IDs,all_possible_paths,shortest_paths,map_info,paths_count_lim=100)
        #储存到字典中
        veh_datas[veh_id]=[paths_possible,paths_w,travel_time,paths_possible_time]
    print("")
    print("初始的可行路径集和权重生成完毕。")

    #然后根据当前veh_datas计算一下每个路段的流量，更新每个路段的行程时间（即cost）
    current_paths = Calc_Current_Paths(veh_datas) #输出每辆车权重最大的那个路径
    node_volumn = Calc_Current_Nodes_volumn(current_paths,cross_ids) #输出当前各个路口的流量
    links_flow,links_cost = Calc_Links_Flow_and_Cost(current_paths,links_info,links_capacity) #计算各个路段的流量大小和行程时间
    re_paths = current_paths #先获取初始解

    #准备工作完成 开始迭代
    # #根据当前最可能的车辆路径计算各路段流量 用于计算误差
    # recon_link_flow,_ = Calc_Link_Flow(current_paths,links_info)
    # #计算当前最可能的车辆路径与真实路径的误差
    # mae,rmse,r2 = Calc_MAE_RMSE_R2(list(recon_link_flow.values()),list(real_link_flow.values()))
    # log = [[mae,rmse,r2]] #记录一下迭代中误差变化情况
    # r2_log = [r2]  #记录一下迭代中R2变化情况
    # print("初始解的路段流量误差为：",f"MAE: {mae:0.1f}, RMSE: {rmse:0.1f}, R2: {r2:0.2f}")

    #迭代次数
    for i in range(iteration_count):

        print("开始进行第{}次迭代...".format(i+1))
        
        #关键迭代计算部分：更新粒子权重
        veh_datas = PF_Iteration(veh_datas,links_flow,links_cost,node_volumn,max_node_volumn)

        current_paths = Calc_Current_Paths(veh_datas) #更新当前迭代结果中 每辆车最可能的出行路径
        current_paths_with_time = Calc_Current_Paths_with_time(veh_datas) #更新当前迭代结果中 每辆车最可能的出行路径(含时间) 这个没什么用 只是用来输出而已

        node_volumn = Calc_Current_Nodes_volumn(current_paths,cross_ids) #更新当前迭代结果中各个路口的流量
        links_flow,links_cost = Calc_Links_Flow_and_Cost(current_paths,links_info,links_capacity) #更新各个路段的流量大小和行程时间
        #输出最终结果
        re_paths = current_paths
        re_paths_with_time = current_paths_with_time
        print("第{}次迭代已完成".format(i+1))


        # #更新完成后评估一下结果
        # recon_link_flow,_ = Calc_Link_Flow(current_paths,links_info) #计算各路段流量
        # mae,rmse,r2 = Calc_MAE_RMSE_R2(list(recon_link_flow.values()),list(real_link_flow.values())) #计算路段流量误差

        # #如果R方结果比之前的好 那就更新结果
        # if r2 >= max(r2_log):
        #     re_paths = current_paths
        #     re_paths_with_time = current_paths_with_time
        # log.append([mae,rmse,r2]) #记录误差
        # r2_log.append(r2)
    
        # print("第{}次迭代已完成，路段流量误差为：".format(i+1),f"MAE: {mae:0.1f}, RMSE: {rmse:0.1f}, R2: {r2:0.2f}")

    return re_paths,re_paths_with_time #,log

#删除指定路口的原始数据
def Gen_Lost_AVI_Datas(datas:DataFrame,lost_avi_ids:list):
    new_datas = datas
    for id in lost_avi_ids:
        new_datas = new_datas[(new_datas["路口ID"] != id) ]

    return new_datas

#统计粒子滤波当前各个路径的结果：这个结果是两个组件联动的关键 
def Calc_PF_Paths_Result(OD_PathSet:dict):
    OD_Paths_Result_PF = {} #格式：{OD：[各个路径的流量]}
    #初始化字典
    for key in OD_PathSet.keys():
        OD_Paths_Result_PF[key] = [0 for _ in range(len(OD_PathSet[key]))]


    return OD_Paths_Result_PF

#生成各个OD之间的可行路径集
def Gen_OD_PathSet(paths_By_PF_Method:list,all_possible_paths:dict,links_info:dict):
    OD_PathSet = {}
    for key in all_possible_paths.keys():
        keys = key.split("-")
        key = (int(keys[0]),int(keys[1]))
        paths = all_possible_paths.get(str(key[0])+"-"+str(key[1]),[])
        if len(paths) == 0:
            continue
        OD_PathSet[key] = paths[:1] #只取一条就行

    for path in paths_By_PF_Method:
        has_wrong_od = False
        O = path[0]
        D = path[-1]
        if O == D: #OD相同就不管了
            continue

        #路径合规检验：可能FCD载客路径中有无法相连的路径 所以要检查一下
        for i in range(len(path)-1):
            od = (path[i],path[i+1])
            if od not in links_info.keys():
                has_wrong_od = True
                break
        
        if has_wrong_od:
            continue

        OD_paths = OD_PathSet.get((O,D),[]) #先去已有路径集中看看
        if OD_paths == []: #如果没有收录 那就给它加进去
            OD_PathSet[(O,D)] = [path]
        else:
            if path not in OD_paths: #如果没有收录 那就给它加进去
                OD_paths.append(path)

    return OD_PathSet

#PEF的内层迭代：
# 算法所需输入：当前路段流量，实际AVI检测流量，路段通行能力，粒子滤波算法当前结果：各个路径的流量
# 实际输入：路段参数数据表，AVI路段的观测结果，可行路径集，粒子滤波算法当前结果，Theta
def PFE_Itera(links_datas:dict,links_flow_avi:dict,OD_PathSet:dict,OD_Paths_Result_PF:dict,Theta = 0.01):
    OD_Paths_Result ={}

    #算法数据
    links_flow = {key:0 for key in links_datas.keys()} #各个路段的流量 初始为0 后续进行更新
    links_cost = {key:0 for key in links_datas.keys()} #各个路段的费用 初始为0 后续进行更新
    links_dual = {key:(0,0,0) for key in links_datas.keys()} #各个路段的对偶变量 [u+,u-,da]
    paths_flow = {} #各路径的流量 初始为零 直接展开路径 不再按照OD字典的方式
    for key in OD_PathSet.keys():
        paths = OD_PathSet[key]
        for path in paths:
            paths_flow[tuple(path)] = 0
    
    paths_flow_PF = {} #粒子滤波估计出的结果 直接展开路径 不再按照OD字典的方式
    for key in OD_PathSet.keys():
        paths = OD_PathSet[key]
        flows = OD_Paths_Result_PF[key]
        for i in range(len(paths)):
            path = paths[i]
            flow = flows[i]
            paths_flow_PF[tuple(path)] = flow

    paths_dual = {key:0 for key in paths_flow.keys()} #各个路径的对偶变量

    #其他算法参数
    obj_ls = [] #记录目标函数(SUE)的当前值
    MAE_links = [] #用来记录每次迭代结果的路段流量误差
    MAE_paths = [] #用来记录每次迭代结果的路径流量误差（与粒子滤波方法相比）
    e_link = 0.06 #路段流量允许的误差
    e_path = 0.3 #路径流量允许的误差
    dua_limit = 1000000000 #路段对偶变量的上下限 即 u_plus 和 u_min 的限制
    step = 0.4 #路径流量每次更新的比例
    k = 0.2 #对偶变量更新的比例
    Stop = 0.2 #迭代收敛限值（对偶变量变化值）
    max_itera_time = 100 #最大迭代次数 20 200


    flag = 100000 #算法收敛标志
    m = 0 #迭代次数
    while flag >= Stop and m <= max_itera_time:
        m += 1 #迭代次数+1

        #step:1 根据上次迭代的路段流量结果 计算各个路段的通行费用 （第一次的时候算出来的是自由流费用）
        for key in links_cost.keys():
            capacity = links_datas[key][2] #读取通行能力
            free_cost = links_datas[key][3] #读取自由流花费
            link_flow = links_flow[key] #读取路段流量
            cost = Calc_Link_Cost(free_cost,capacity,link_flow) #计算新费用
            links_cost[key] = cost #记下新费用

        #step:2 计算目标函数(SUE) 更新结果
        #第一项 路径选择情况的对数计算
        f_temp = [(flow * (math.log(flow)-1)) if flow > 0 else 0 for flow in paths_flow.values()] #注意0不能取对数
        obj_ls.append(sum(links_cost.values())+sum(f_temp)/Theta) #在列表里记录下目标函数的值 便于后期分析迭代情况


        #step:3 计算新的路径流量
        paths_flow_temp = {key:0 for key in paths_flow.keys()} #新建临时的路径流量表 全为0
        #遍历每一条路径
        for path in paths_flow_temp.keys(): #字典的key就是具体路径
            #路径的费用
            path_cost = 0
            #获取路段的对偶变量
            u_plus = 0
            u_min = 0
            da = 0
            #解析路径 拆分成一个个路段 累计计算上述参数
            for i in range(len(path)-1):
                o = path[i]
                d = path[i+1]
                u_plus += links_dual[(o,d)][0] #累加u+
                u_min += links_dual[(o,d)][1] #累加u-
                da += links_dual[(o,d)][2] #累加da
                path_cost += links_cost[(o,d)] #累加路径费用

            #计算路径流量的变化
            paths_flow_temp[path] = math.exp(Theta*(u_plus + u_min + da + paths_dual[path] - path_cost))
        #更新出新的路径流量
        paths_flow = {key:(value+step*( paths_flow_temp[key]-value)) for key,value in paths_flow.items()}


        #step:4 由路径流量反推,重新计算各个路段的流量
        links_flow = {key:0 for key in links_datas.keys()} #各个路段的流量重置为0
        #遍历每一条路径，把路段流量加起来得到总的路段流量
        for path,flow in paths_flow.items():
            for i in range(len(path)-1):
                o = path[i]
                d = path[i+1]
                links_flow[(o,d)] += flow


        #step:5 更新路段对偶变量
        #先记录一下路段对偶变量 这个以后需要新旧相减来判断是否结束迭代
        links_dual_old = links_dual.copy()
        for key in links_dual.keys():
            #区分有AVI的路段和无AVI的路段 它们要用的对偶变量不一样  (如果这个路段上没有流量 就不需要更新)
            has_avi = links_datas[key][0]

            if has_avi and links_flow[key] > 0 : #注意：要求路段上有流量 否则除以0出错

                u_plus = links_dual[key][0] + k * math.log((1 + e_link) * (links_flow_avi[key]/links_flow[key]))/Theta # dual(i,1)+k*log((1+thlea)*gridlink(i,6)/linkflow(i))/deta;
                u_plus = min(0,u_plus)

                if u_plus < -dua_limit: #不要超过下限
                    u_plus = -dua_limit

                u_min = links_dual[key][1] + k * math.log((1 - e_link) * (links_flow_avi[key]/links_flow[key]))/Theta # dual(i,2)+k*log((1-thlea)*gridlink(i,6)/linkflow(i))/deta;
                u_min = max(0,u_min)

                if u_min > dua_limit: #不要超过上限
                    u_min = dua_limit
            
                #将更新结果记录下来
                links_dual[key] = (u_plus,u_min,0)

            elif (not has_avi) and links_flow[key] > 0:  #注意：要求路段上有流量 否则除以0出错
                
                da = links_dual[key][2] + k * math.log((links_datas[key][2]/links_flow[key]))/Theta # dual(i,3)+k*log(gridlink(i,4)/linkflow(i))/deta;
                da = min(0,da)

                if da < -dua_limit: #不要超过下限
                    da = -dua_limit

                #将更新结果记录下来
                links_dual[key] = (0,0,da)


        #step:6 根据推出来的路段流量 重新计算路段费用（与step:1完全一致）
        for key in links_cost.keys():
            capacity = links_datas[key][2] #读取通行能力
            free_cost = links_datas[key][3] #读取自由流花费
            link_flow = links_flow[key] #读取路段流量
            cost = Calc_Link_Cost(free_cost,capacity,link_flow) #计算新费用
            links_cost[key] = cost #记下新费用


        #step:7 因为有了新的对偶变量，那么可以再次更新计算路径流量（与step:3完全一致）
        paths_flow_temp = {key:0 for key in paths_flow.keys()} #新建临时的路径流量表 全为0
        #遍历每一条路径
        for path in paths_flow_temp.keys(): #字典的key就是具体路径
            #路径的费用
            path_cost = 0
            #获取路段的对偶变量
            u_plus = 0
            u_min = 0
            da = 0
            #解析路径 拆分成一个个路段 累计计算上述参数
            for i in range(len(path)-1):
                o = path[i]
                d = path[i+1]
                u_plus += links_dual[(o,d)][0] #累加u+
                u_min += links_dual[(o,d)][1] #累加u-
                da += links_dual[(o,d)][2] #累加da
                path_cost += links_cost[(o,d)] #累加路径费用

            #计算路径流量的变化
            paths_flow_temp[path] = math.exp(Theta*(u_plus + u_min + da + paths_dual[path] - path_cost))
        #更新出新的路径流量：解可能是小数
        # paths_flow = {key:(value+step*( paths_flow_temp[key]-value)) for key,value in paths_flow.items()}
        #只接受整数结果：四舍五入
        paths_flow = {key:round(value+step*( paths_flow_temp[key]-value)) for key,value in paths_flow.items()}


        #step:8 更新路径对偶变量
        #先记录一下路径对偶变量 这个以后需要新旧相减来判断是否结束迭代
        paths_dual_old = paths_dual.copy()
        for key in paths_dual.keys():
            if paths_flow[key] > 0 and paths_flow_PF[key] > 0:

                paths_dual[key] = paths_dual[key] + k * math.log((1-e_path)*(paths_flow_PF[key]/paths_flow[key]))/Theta  # dualpath(i)+k*log((1-e)*repathflow(i)/pathflow(i))/deta;
                paths_dual[key] = max(0,paths_dual[key])

                if paths_dual[key] > dua_limit: #不要超过上限
                    paths_dual[key] = dua_limit


        #step:9 更新本次迭代的路段流量结果（与step:4完全一致）
        links_flow = {key:0 for key in links_datas.keys()} #各个路段的流量重置为0
        #遍历每一条路径，把路段流量加起来得到总的路段流量
        for path,flow in paths_flow.items():
            for i in range(len(path)-1):
                o = path[i]
                d = path[i+1]
                links_flow[(o,d)] += flow


        #step:10 结束 计算收敛指标 顺便看看与AVI观测结果的MAE（路段流量平均误差）
        #计算迭代前后路段对偶参数的差值绝对值 的最大值
        links_dual_delta_max = max([max([abs(links_dual[key][0]-links_dual_old[key][0]),abs(links_dual[key][1]-links_dual_old[key][1]),abs(links_dual[key][2]-links_dual_old[key][2])]) for key in links_dual.keys()])
        #计算迭代前后路径对偶参数的差值绝对值 的最大值
        paths_dual_delta_max = max([abs(paths_dual[key]-paths_dual_old[key]) for key in paths_dual.keys()])
        #收敛指标计算：两种对偶参数差值取最大值
        flag = max(links_dual_delta_max,paths_dual_delta_max)

        #最后计算一下当前结果与AVI观测结果的MAE（路段流量平均误差）
        MAE = 0
        count = 0
        #只计算有AVI的路段 并且路段流量不能太小
        for key in links_flow.keys():
            has_avi = links_datas[key][0]
            if has_avi:
                link_flow_avi = links_flow_avi[key]
                link_flow_PFE = links_flow[key]
                if link_flow_avi >0 and link_flow_PFE >0:  #500  暂时先不设限制
                    MAE += abs(link_flow_avi - link_flow_PFE)
                    count += 1

        MAE = MAE/count if count > 0 else 0

        #记录一下MAE的结果
        MAE_links.append(MAE)
        
        
        #最后计算一下当前结果与粒子滤波结果的误差（路径流量平均误差） #只统计流量大一些的路径
        MAE = 0
        count = 0
        for key in paths_flow.keys():
            path_flow_PF = paths_flow_PF[key]
            path_flow_PFE = paths_flow[key]

            if path_flow_PFE >= 0: #50 暂时先不设限制
                MAE += abs(path_flow_PF - path_flow_PFE)
                count += 1

        MAE = MAE/count if count > 0 else 0

        #记录一下MAE的结果
        MAE_paths.append(MAE)


    #最后迭代完成后导出结果
    OD_Paths_Result = paths_flow

    return OD_Paths_Result,links_flow,obj_ls,MAE_links,MAE_paths

#路径重构算法：PFE路径流量估计器方法，使用 FCD载客路径集 + 全路网的路段流量 来重构车辆路径
def Generate_PFE_Paths(FCD_paths_set:list,links_flow_avi:dict,links_info:dict,links_capacity:dict):

    #读取任意两个OD之间的可能路径集 后边要用到
    all_possible_paths = np.load('inputs\\all_possible_paths_10_5.npy',allow_pickle=True).item() #预制的任意两OD之间的可行路径集 注意key是str的形式 不是tuple

    #生成各个OD之间的可行路径集
    OD_PathSet = Gen_OD_PathSet(FCD_paths_set,all_possible_paths,links_info)



    #开始路径重构
    #1.1 路径流量估计器初始化
    #1.1.1 参数准备：构建一个路段信息字典 把所有需要用到的和路段相关的参数都放进去
    links_datas = {} #[0:是否为AVI路段,1:对偶变量[u+,u-,da],2:通行能力,3:自由流费用,4:当前流量,5:当前费用]
    for key in links_info.keys():
        data = []
        #0.记录该路段是否为AVI路段
        if key in links_flow_avi.keys(): #如果这个路段是AVI观测路段 记为true；非观测路段记为false
            data.append(True)
        else:
            data.append(False)
        
        #1.对偶变量初始化 全都是0 (AVI路段对偶变量用的是u+,u-，非AVI路段用的是da)
        data.append([0,0,0])

        #2.通行能力
        capacity = max(links_capacity[key]*0.8,2400) #links_capacity是根据流量统计出来的，暂时先乘个0.8的系数，以后直接按照车道数来规定每个路段的通行能力 默认通行能力2400
        data.append(capacity)

        #3.自由流费用
        distance = links_info[key]["长度"] #单位：米
        speed_limit = links_info[key]["限速"]/3.6 #限速 单位 m/s
        free_cost = distance/speed_limit  #默认的通行时间是路段长度除以限速 单位：秒
        data.append(free_cost)


        #4.当前流量 初始为0
        data.append(0)

        #5.当前费用 初始为自由流速度
        data.append(free_cost)

        #记录下来
        links_datas[key] = data



################################2021.11.08测试内容：使用完整OD_PathSet进行PFE#########################################
    #保存一下PFE的工作路径集
    # np.save("outputs\\OD_PathSet.npy",OD_PathSet)
    # exit()
    # #直接加载OD_PathSet作为工作路径集，看看效果是怎么样
    # OD_PathSet = np.load("outputs\\OD_PathSet.npy",allow_pickle=True).item()
################################2021.11.08测试内容：使用完整OD_PathSet进行PFE#########################################


    #1.2 开始迭代，求出PFE的解
    #先获取粒子滤波的结果
    OD_Paths_Result_PF = Calc_PF_Paths_Result(OD_PathSet)
    #开始迭代求解
    OD_Paths_Result,links_flow,obj_ls,MAE_links,MAE_paths = PFE_Itera(links_datas,links_flow_avi,OD_PathSet,OD_Paths_Result_PF)

    print("PFE Done.")
    return OD_Paths_Result,links_flow,obj_ls,MAE_links,MAE_paths