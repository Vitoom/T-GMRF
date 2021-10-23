#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: point.py
Author: K
Email: 7thmar37@gmail.com
Github: https://github.com/7thMar
Description: 输出所有数据集的分布点
"""

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as f
cnfont = f.FontProperties(fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False
# plt.title('中文示例', fontproperties=cnfont)


if __name__ == '__main__':
    fig, axes = plt.subplots(5, 2, figsize=(12, 30))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.03, hspace=0.2, wspace=0.2)

    path = sys.path[0] + '/dataset/'
    files = os.listdir(path)
    i, j = 0, 0
    for file in files:
        data_path = path + file
        data_title = os.path.splitext(file)[0]
        points = pd.read_csv(data_path, sep='\t', usecols=[0, 1])
        axes[i][j].scatter(points.loc[:, 'x'], points.loc[:, 'y'], s=1)
        axes[i][j].set_title(data_title)
        if j == 1:
            i, j = i + 1, 0
        else:
            j += 1

    plt.show()
