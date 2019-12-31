#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: wyb Time:2019/11/16
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN
from kmedoid import kMedoids
from mpl_toolkits.basemap import Basemap
from sqlalchemy import create_engine
from sqlalchemy.sql import text

def ship_sort(data):
    # 将数据按postime排序
    data = data.sort_values(by='postime')
    # 根据mmsi分组
    dataGroup = data.groupby('mmsi')

    L = []
    # 将每一条船按照时间戳连起来
    for name, group in dataGroup:  # name 指mmis， group指dataFrame
        xyList = [xy for xy in zip(group.lon, group.lat)]
        if len(xyList) > 1:
            L.append(np.array(xyList))
    return L


def plot_cluster(traj_lst, cluster_lst):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    cluster_count = np.max(cluster_lst) + 1
    m = Basemap(llcrnrlon=117.339, llcrnrlat=23.539, urcrnrlon=117.663, urcrnrlat=23.913,
                resolution='h', projection='mill')
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='#ddaa66', lake_color='aqua')
    m.drawcoastlines()
    for traj, cluster in zip(traj_lst, cluster_lst):
        # x, y = traj[:, 0], traj[:, 1]
        x, y = m(traj[:, 0], traj[:, 1])
        if cluster == -1:
            # Means it it a noisy trajectory, paint it black
            #plt.plot(x, y, c='k', linestyle='dashed')
            pass
        else:
            plt.plot(x, y, c=color_lst[cluster % len(color_lst)])
    plt.show()


def hausdorff(u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d


def my_cluster(data, methods='kmedois', k=6, eps=0.15, min_samples=6):
    # 将每条船按postime顺序排序
    traj_lst = ship_sort(data)
    traj_count = len(traj_lst)
    D = np.zeros((traj_count, traj_count))
    # 计算距离矩阵
    for i in range(traj_count):
        for j in range(i + 1, traj_count):
            distance = hausdorff(traj_lst[i], traj_lst[j])
            D[i, j] = distance
            D[j, i] = distance
        # 4 - Different clustering methods
    if methods == 'kmedois':
        # 4.1 - kmedoids
          # The number of clusters
        medoid_center_lst, cluster2index_lst = kMedoids(D, k)
        cluster_lst = np.empty((traj_count,), dtype=int)
        for cluster in cluster2index_lst:
            cluster_lst[cluster2index_lst[cluster]] = cluster

    if methods == 'dbscan':
        # 4.2 - dbscan
        mdl = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_lst = mdl.fit_predict(D)

    return traj_lst, cluster_lst


if __name__ == '__main__':
    # Some visualization stuff, not so important
    sns.set()
    plt.rcParams['figure.figsize'] = (12, 12)
    # Utility Functions
    color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                      'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])
    # 读取数据
    conn = create_engine('mysql+pymysql://root:root123@localhost:3306/gulei')
    sql = "select  mmsi,cast(lat as real) lat,cast(lon as real) lon,postime  from trackdata where line='bottom2top' order by mmsi  "
    data = pd.read_sql(sql, conn)
#    data = pd.read_csv('data.csv', encoding='gbk')
    # 对船的轨迹进行聚类 methods = {'kmedois', 'dbscan'}
    traj_lst, cluster_lst = my_cluster(data, methods='dbscan')
    # 可视化
    plot_cluster(traj_lst, cluster_lst)

