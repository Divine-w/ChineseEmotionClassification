# 中文文本情感分类（基于LSTM和textCNN）
数据文本都是用爬虫从网络上爬取的，由人工进行分类，data文件下存储的是已经经过预处理的数据，模型嵌入层部分使用了预训练的中文词向量，因为文件比较大需要自己下载放入data文件夹下，这里给出中文词向量项目地址：[https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)，本项目使用的是用微博语料数据训练出的300维的词向量。
## 依赖库
 - python 3.7.5
 - sklearn 0.22.1
 - pytorch 1.2.0
 - torchtext 0.6.0
 - tqdm 4.40.0
## 文件说明
 - biLSTM.py 双向LSTM实现
 - textCNN.py TextCNN实现
 - models.py 存储模型类
 - utils.py 工具库
 - data
   - negative_process.data 预处理过的负面文本
   - positive_process.data 预处理过的正面文本

**博客地址：**[https://blog.csdn.net/Divine0/article/details/106639492](https://blog.csdn.net/Divine0/article/details/106639492)
