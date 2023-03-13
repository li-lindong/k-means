# --coding: utf-8--
# k_means.py

import torch
import random

class Kmeans:
    def __init__(self, k=2, tolerance=0.00001, max_iter=300):
        """
        :param k: 整数类型，表示簇的数量，即，将输入的特征等聚为多少组
        :param tolerance: 前后聚类中心的距离的边界值，如果都小于边界值，则聚类结束
        :param max_iter: 最大迭代次数，达到该迭代次数，不管其它条件，都结束聚类
        """
        self.k = k
        self.tol = tolerance
        self.max_iter = max_iter
        self.features_count = -1
        self.classifications = None     # 属于同一类的数据放在一个列表中
        self.centroids = None      # 聚类中心

    def fit(self, data):
        """
        :param data: tensor数组，约定shape为：(数据数量，数据维度)；需要聚类是数据
        :type data: torch.tensor
        """
        self.features_count = data.shape[1]     # 输入数据的特征个数
        # 随机选择k个数据作为初始化的聚类中心（维度：k个 * features种数）
        self.centroids = torch.zeros([self.k, data.shape[1]])
        index = [random.randint(0, data.shape[0]) for i in range(self.k)]
        self.centroids = data[index]

        for i in range(self.max_iter):
            # 清空聚类列表
            self.classifications = [[] for i in range(self.k)]
            # 对每个数据与聚类中心进行距离计算
            for feature_set in data:
                # 预测各数据属于哪一分类
                classification = self.predict(feature_set)
                # 将数据加入对应的类别
                self.classifications[classification].append(feature_set)

            # 记录前一次的聚类中心
            prev_centroids = self.centroids.clone()

            # 根据新的聚类结果更新中心，这里采用类内所有数据的逐点平均作为聚类中心
            for classification in range(self.k):
                self.centroids[classification] = torch.mean(torch.stack(self.classifications[classification]), dim=0)

            # 检测前后两次中心的变化情况，如果都小于等于边界值，则停止计算，返回
            for c in range(self.k):
                if torch.dist(prev_centroids[c], self.centroids[c], p=2) > self.tol:
                    break   # 如果存在大于边界值的情况，则继续迭代
            # 如果都满足条件（上面循环没break），则返回
            else:
                return

    def predict(self, data):
        """
        :param data: 输入的一个数据
        :return: 最小距离索引，即该数据与哪一个聚类中心最近，则属于该聚类中心对应的簇
        """
        # 计算当前数据与各聚类中心的距离
        distances = [torch.dist(data, i, p=2) for i in self.centroids]
        # 返回最小距离索引
        return distances.index(max(distances))

if __name__ == '__main__':
    input = torch.rand(10, 512)
    model = Kmeans()
    model.fit(input)