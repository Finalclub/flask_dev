#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# @Version : 1.1
# @Time    : 2019/4/19 20:15
# @Author  : zhaidongchang
# @File    : distanceMetricLSH.py
# @Revised : v1.1将哈希对象转换为像素

import numpy as np
import hashtable as ht


class DMLSH(object):
    """docstring for HashTable"""

    def __init__(self, parameterK, parameterR, parameterL):

        # parameter: k
        self.parameterK = parameterK
        # parameter: r
        self.parameterR = parameterR
        # parameter: L
        self.parameterL = parameterL
        # parameter: C
        self.C = np.power(2, 32) - 5

    def funKHashFamilyGenerator(self, inputDimmmension):
        '''
        根据公式 h(v) = (a * v + b) / r
        parameterR即长度r
        '''
        # 生成形为(k, input_dim)的hash方程矩阵，计算k个hash值
        hashFunctionsMatric = np.random.randn(self.parameterK, inputDimmmension)
        # 生成参数b，形为(k,1)
        b = np.asarray([np.random.uniform(0, self.parameterR) for _ in range(self.parameterK)]).reshape(self.parameterK, 1)
        # 以列表形式返回生成的一对[h(·),b]
        return [hashFunctionsMatric, b]

    def funCalculateHashValues(self, inputKHashfamily, inputVector):
        # 计算k个hash值，return kHashValues 的形为(kHashValues ,1)
        return (np.dot(inputKHashfamily[0], inputVector.reshape(-1, 1)) + inputKHashfamily[1]) // self.parameterR

    def funFingerPrint(self, randomValueForFP, kHashValues):
        # 定义指纹计算函数，返回一个整数
        return int(np.dot(randomValueForFP, kHashValues) % self.C)

    def funDistanceMetricLSHIndex(self, inputDataSet, inputPattern):
        '''
        输入数据集进行索引
        Pixel模式用于算法实验
        Bands模式用于测试LSH算法的效果

        '''
        # 调用hashTable生成L个hash tables
        self.hashTables = [ht.HashTable(i) for i in range(self.parameterL)]
        # 生成计算指纹的随机常数
        self.randomValueForFP = np.random.randint(-10, 10, (1, self.parameterK))
        # 进入Pixel模块
        if inputPattern == 'Pixel':
            # 将L个 g(·) 以list 的方式存储，输入的数据为波段个数
            numOfFeatures = inputDataSet.shape[1]
            self.LHashGroups = [self.funKHashFamilyGenerator(numOfFeatures) for _ in range(self.parameterL)]
            # 将L个hash_function和 特征(矩阵列数）个向量相乘，计算得到指纹和索引，将数据存放在相应的索引
            for group in self.LHashGroups:
                for i, vector in enumerate(inputDataSet):
                    # 计算k hashvalues
                    kHashValuesArray = self.funCalculateHashValues(group, vector)
                    fingerPrint = self.funFingerPrint(self.randomValueForFP, kHashValuesArray)
                    hashIndex = fingerPrint % self.parameterL
                    # hashTables[hashIndex].buckets为读取的字典，存放{fingerPrint:vector_index}
                    if fingerPrint in self.hashTables[hashIndex].buckets:
                        self.hashTables[hashIndex].buckets[fingerPrint].append(i)
                    else:
                        self.hashTables[hashIndex].buckets[fingerPrint] = [i]
            # 此处可以Redis 存储
            # return self.hashTables, self.LHashGroups
        # 采用波段进行哈希时，需要将输入的矩阵转置
        elif inputPattern == 'Band':
            # 将L个 g(·) 以list 的方式存储，输入的数据为样本个数
            numOfpixels = inputDataSet.shape[0]
            self.LHashGroups = [self.funKHashFamilyGenerator(numOfpixels) for _ in range(self.parameterL)]
            # 将L个hash_function和 特征(矩阵列数）个向量相乘，计算得到指纹和索引，将数据存放在相应的索引
            for group in self.LHashGroups:
                # testpart
                for i in range(inputDataSet.shape[1]):
                    kHashValuesArray = self.funCalculateHashValues(group, inputDataSet[:, i:i+1])
                    fingerPrint = self.funFingerPrint(self.randomValueForFP, kHashValuesArray)
                    hashIndex = fingerPrint % self.parameterL
                    # hashTables[hashIndex].buckets为读取的字典，存放{fingerPrint:vector_index}
                    if fingerPrint in self.hashTables[hashIndex].buckets:
                        self.hashTables[hashIndex].buckets[fingerPrint].append(i)
                    else:
                        self.hashTables[hashIndex].buckets[fingerPrint] = [i]
            # self.hashTables = hashTables
            # 此处可以Redis 存储
            # return self.hashTables, self.LHashGroups
        else:
            print('Please input corrected pattern.')

    def funNNSearch(self, queryVector):
        # 存储查询结果
        QueryResults = set()

        for hashFunctions in self.LHashGroups:
            queryHashvalue = self.funCalculateHashValues(hashFunctions, queryVector)
            queryFingerPrinter = self.funFingerPrint(self.randomValueForFP, queryHashvalue)
            queryIndex = queryFingerPrinter % self.parameterL
            if queryFingerPrinter in self.hashTables[queryIndex].buckets:
                QueryResults.update(self.hashTables[queryIndex].buckets[queryFingerPrinter])

        return QueryResults
