#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib

reg = joblib.load('/home/zhang/jupyterProject/solution/reg.model')
poly = joblib.load('/home/zhang/jupyterProject/solution/poly.model')
before_poly = np.load('/home/zhang/jupyterProject/solution/before_poly.npy')
p = np.load('/home/zhang/jupyterProject/solution/p.npy')
fips_ = np.load('/home/zhang/jupyterProject/solution/fips_.npy')
D = np.fromfile("/home/zhang/jupyterProject/solution/D.bin", dtype=np.float64).reshape(461,461)
MCM_H_drop_loc = pd.read_csv('/home/zhang/jupyterProject/solution/MCM_H_drop_loc.csv')
MCM_F_drop_loc = pd.read_csv('/home/zhang/jupyterProject/solution/MCM_F_drop_loc.csv')
plt.rcParams['figure.figsize'] = (16.0, 10.0)
plt.rc('font', size=18)
fig, ax = plt.subplots()

# x = np.arange(0,10,0.1)
# line, = ax.plot(x,np.sin(x))

# def animat(i):
#     line.set_ydata(np.sin(x+i/100))
#     return line,
# def initial():
#     line.set_ydata(np.cos(x))
#     return line,
# ani = animation.FuncAnimation(fig=fig,func=animat,frames=100,init_func=initial,interval=20,blit=False)


map_ = Basemap(projection='stere', lat_0=90, lon_0=-105, \
               llcrnrlat=MCM_H_drop_loc.lon.min(), urcrnrlat=MCM_H_drop_loc.lon.max() - 1, \
               llcrnrlon=MCM_H_drop_loc.lat.min() - 0.5, urcrnrlon=MCM_H_drop_loc.lat.max() + 5, \
               rsphere=6371200., resolution='h', area_thresh=10000, ax=ax)

map_.drawmapboundary()  # 绘制边界
map_.drawstates()  # 绘制州
map_.drawcoastlines()  # 绘制海岸线
map_.drawcountries()  # 绘制国家
# map.drawcounties()      # 绘制县

parallels = np.arange(0., 90, 10.)
map_.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=20)  # 绘制纬线

meridians = np.arange(-110., -60., 10.)
map_.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=20)  # 绘制经线
# MCM_F_drop_loc.YYYY =
posi = MCM_H_drop_loc[MCM_H_drop_loc.YYYY == 2017]
posi = posi.sort_index(by=['FIPS_Combined'], ascending=True)


# posi = MCM_H_drop_loc[MCM_H_drop_loc.YYYY == 2010]
# posi=pd.read_csv("2014_us_cities.csv") # 读取数据


lat = np.array(posi["lon"][:])  # 获取维度之维度值
lon = np.array(posi["lat"][:])  # 获取经度值
pop = np.array(posi["DrugReports"][:], dtype=float)  # 获取海洛因数，转化为numpy浮点型

# 通过对比预测与实际的曲线来看看效果----------
# collect_predict_1 = [pop.tolist()]


size = (pop / np.max(pop)) * 1000  # 绘制散点图时图形的大小，如果之前pop不转换为浮点型会没有大小不一的效果
x, y = map_(lon, lat)

# map_.scatter(x,y,s=size)     #
# plt.title('Predict distribution of drugs in 2013')
pop_ = pop.copy()

###  222222222222
posi = MCM_F_drop_loc[MCM_F_drop_loc.YYYY == 2015]
posi = posi.sort_index(by=['FIPS_Combined'], ascending=True)
# pop2 = np.array(posi["DrugReports"][:], dtype=float)
# size2 = (pop2 / np.max(pop2)) * 1000  # 绘制散点图时图形的大小，如果之前pop不转换为浮点型会没有大小不一的效果
# x2, y2 = map_(lon+np.random.rand(*lon.shape)/10, lat+np.random.rand(*lon.shape)/10)
# a2 = map_.scatter(x2, y2, s=size2, ax=ax, c='green', alpha=0.6)
a = map_.scatter(x, y, s=size, ax=ax, alpha=0.8)
# pop2_ = pop2.copy()

# collect_predict_2 = [pop2.tolist()]
# plt.legend(['Fentanyl', 'Heroin'])
# a.remove()
# reg.preict

plt.ion()
fips_source = [39061, 39113, 39041, 42003]
fips = np.unique(MCM_H_drop_loc.FIPS_Combined.values).tolist()

state_info = np.array(list(map(lambda t: t//1000, fips)))

## recode
# predict = MCM_F_drop_loc.copy()
## --
for time in range(100):
    # for source in fips_source:
    #     source_id = fips.index(source)
    #     pop[source_id] += 0.1
        # pop2[source_id] += 0.1
    for i in range(len(fips)):
        # delta_pop = np.clip(pop_ - pop_[i], a_min=0,
        #                     a_max=np.inf)  # Only use the other party more than yourself will be affected

        # delta_pop2 = np.clip(pop2_ - pop2_[i], a_min=0,
        # #                     a_max=np.inf)
        # delta_D = D[i]  # +
        # delta = delta_pop / (delta_D+1)
        # delta = delta_pop * np.exp(-delta_D / 1000)
        # delta = delta_pop * delta_D.max()/(delta_D+1)/10000
        delta = p[fips_==fips[i]]
        # delta2 = delta_pop2 / (delta_D+1)
        # delta2 = delta_pop * np.exp(-delta_D / 1000)
        # delta2 = delta_pop2 * delta_D.max()/(delta_D+1)/10000
        pop[i] += sum(delta)
        # pop2[i] += sum(delta2)

    pop_ = pop
    # pop2_ = pop2
    if pop[state_info == 21].sum() >= 439.5*100:
        print(21)
        break
    if pop[state_info == 39].sum() >= 1157*100:
        print(39)
        break
    if pop[state_info == 42].sum() >= 1277.3*100:
        print(42)
        break
    if pop[state_info == 51].sum() >= 826*100:
        print(51)
        break
    if pop[state_info == 54].sum() >= 185.4*100:
        print(54)
        break


    # collect_predict_1.append(pop.tolist())
    # collect_predict_2.append(pop2.tolist())
    # rank = predict[predict.YYYY == (1+time + 2010)].FIPS_Combined.rank().values.astype(int)
    # predict.loc[predict.YYYY == (1+time + 2010), 'DrugReports'] = pop2_[rank - 1]

    plt.pause(0.01)
    size = (pop / np.max(pop)) * 1000
    # size2 = (pop2 / np.max(pop2)) * 1000
    a.remove()
    # a2.remove()


    # a2 = map_.scatter(x2, y2, s=size2, ax=ax, c='green', alpha=0.6)
    a = map_.scatter(x, y, s=size, ax=ax, alpha=0.8)
    # plt.show()
    # print(time)

plt.savefig('break out!!!!!.svg')
plt.show()
# plt.figure()
# plt.plot(MCM_H_drop_loc.YYYY==2010)
# plt.figure()
# plt.plot(list(collect_predict_1))
# plt.plot(list(zip(*collect_predict_1)))
# plt.figure()
# plt.plot(list(zip(*collect_predict_1)))

# predict.to_csv('predict2.csv')
pause()
