# coding: utf-8
#读取FCD行驶路径（node-node形式），计算各路段的行程时间和FCD路段流量
#FCD行驶路径文件：inputs\\FCD_Paths\\{date}_{st}-{et}.npy



import numpy as np


#Step 1:准备工作
#读取FCD数据文件
datetime = "20210106_10-11" #"20210106_6-7"
FCD_Paths = np.load("inputs\\FCD_Paths\\"+ datetime +".npy",allow_pickle=True).item()


#Step 2:计算各路段的行程时间
#创建一个字典，存储各路段行程时间结果
travel_time = {}

#遍历每一条出行路径
for path in FCD_Paths.values():
    #如果记录的路径节点长度小于2，那就跳过这条路径
    if len(path) < 2:
        continue
    #依次截取相邻两个节点数据
    for i in range(len(path)-1):
        #获取路口ID
        o = path[i][0]
        d = path[i+1][0]
        #如果两个路口ID相同，就跳过这段数据
        if o == d:
            continue
        #获取车辆经过这两个路口的时间字符串
        time_o = path[i][1]
        time_d = path[i+1][1]
        #如果这两个字符串有任何一个是空的，那就跳过
        if time_o == "" or time_d == "":
            continue
        #如果两个时间都不为空 那就解析时间
        #解析小时
        time_o_h = time_o[11:13]
        time_d_h = time_d[11:13]
        #解析分钟
        time_o_m = time_o[14:16]
        time_d_m = time_d[14:16]
        #解析秒钟
        time_o_s = time_o[17:19]
        time_d_s = time_d[17:19]
        #时间转换为秒数
        time_o = int(time_o_h)*3600 + int(time_o_m)*60 + int(time_o_s)
        time_d = int(time_d_h)*3600 + int(time_d_m)*60 + int(time_d_s)
        #相减计算行程时间
        delta_t = time_d-time_o
        #时间间隔不应该小于10秒 因为两个FCD数据点的采集时间差就是10秒
        if delta_t < 10: 
            continue
        #把计算好的行程时间记录到字典里
        temp = travel_time.get((o,d),[])
        temp.append(delta_t)
        travel_time[(o,d)]=temp

#最后保存行程时间字典
np.save("outputs\\FCD_Travel_Time\\"+ datetime +".npy",travel_time)


#Step 3:计算各路段的FCD流量
#创建一个字典，存储各路段FCD流量结果
FCD_Links_Flow = {}
#遍历每一条出行路径
for path in FCD_Paths.values():
    #如果记录的路径节点长度小于2，那就跳过这条路径
    if len(path) < 2:
        continue
    #如果两个路口ID相同，就跳过这段数据
    if o == d:
        continue
    #依次截取相邻两个节点数据
    for i in range(len(path)-1):
        #获取路口ID
        o = path[i][0]
        d = path[i+1][0]
        #流量累计
        FCD_Links_Flow[(o,d)] = FCD_Links_Flow.get((o,d),0) + 1

#最后保存路段FCD流量字典
np.save("outputs\\FCD_Links_Flow\\"+ datetime +".npy",FCD_Links_Flow)