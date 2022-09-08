"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/3/14 21:26
"""
import numpy as np
from collections import Counter
from typing import List, Dict


class BM25Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        # 文本列表，内部每个文本需要事先分好词
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 文本库中文本的平均长度
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        # 存储每个文本中每个词的词频
        self.f = []
        # 存储每个词汇的逆文档频率
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        # 类初始化
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                    self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                             qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list


class BM25ModelV2:
    def __init__(self, doc_word_frequency: List[Dict[str, int]], k1=2, k2=1, b=0.5):
        self.doc_word_frequency = doc_word_frequency
        self.documents_number = len(doc_word_frequency)
        self.avg_documents_len = sum([sum(doc.values()) for doc in doc_word_frequency]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for doc_word_freq in self.doc_word_frequency:
            self.f.append(doc_word_freq)
            for key in doc_word_freq.keys():
                df[key] = df.get(key, 0) + 1
            del doc_word_freq
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                    self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                             qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list
