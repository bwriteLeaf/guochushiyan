import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
from pyecharts import Pie
from pyecharts_snapshot.main import make_a_snapshot

class FigureHelper:
    def __init__(self):
        self.gernerateFigure = self.gernerateFigureWrapper()
        matplotlib.rcParams['font.sans-serif'] = 'Microsoft YaHei'

    # 生成一个图表，返回其ID
    def gernerateFigureWrapper(self):  # 第一个参数为用户指定的图片大小,如 [8, 4.8]
        figureID = 0

        def func():
            nonlocal figureID

            plt.figure(str(figureID))
            figureID += 1
            return str(figureID - 1)

        return func

# 绘制分组线图
    # dataList: 待绘制的数据集，为小数
    # dataLabelList: 与dataList相对应的标签，被用作图例
    # xAxisLabelList: x轴数据标签
    # xLable x轴标签
    # yLable y轴标签
    def lineChartPlot(self,dataList, dataLabelList, xAxisLabelList, xLable=None, yLable=None):
        figureId = self.gernerateFigure()
        legends = []
        if len(dataList) == 0:
            return None

        case_cnt = len(dataList)

        for i in range(0, case_cnt):
            xs = xAxisLabelList
            ys = dataList[i]
            l = plt.plot(xs, ys, label=dataLabelList[i])
            legends.append(l)

        if case_cnt >= 2:
            plt.legend()

        if yLable is not None:
            plt.ylabel(yLable)
        if xLable is not None:
            plt.xlabel(xLable)


        return figureId


    def linePlot(self,dataList, dataLabelList, xAxisLabelList,picPath):
        id = self.lineChartPlot(dataList, dataLabelList, xAxisLabelList)
        f = plt.figure(id)
        f.savefig(picPath)

if __name__ == '__main__':
    figureHelper = FigureHelper()
    dataList = [[0.11, 0.26, 0.13, 0.32, 0.44, 1.16], [0.11, 0.23, 0.13, 0.24, 0.31, 0.7]]
    dataLabelList = ["a","b"]
    xAxisLabelList = [1, 2, 5, 7, 9, 11]
    figureHelper.linePlot(dataList, dataLabelList, xAxisLabelList,"newtest")

