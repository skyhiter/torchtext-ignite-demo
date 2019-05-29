## Pytorch torchtext and ignite demo

`torchtext`与`ignite`是`pytorch`官方提供的模块，前者负责数据的预处理、生成dataloader等，后者负责训练循环控制。本仓库是通过简单的情感分类任务来练习如何使用这个两个模块，其中模型为`BiLSTM`，数据集为`MR dataset`。本代码可自动检测`NVIDIA`显卡(`CUDA`)，如果有`CUDA`设备则使用`GPU`加速训练，否则使用`CPU`训练。

### 实验语料
本`demo`使用的是经典的MR(Movie Review)情感分析语料，共10662条句子。官网为http://www.cs.cornell.edu/people/pabo/movie-review-data , 需下载其中的`sentence polarity dataset v1.0`。

#### 下载链接
[Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)

### Pre-Trained词向量
本`demo`默认使用的是`Pre-Trained GloVe Word Embedding`, 版本为`glove.6B.50d.txt`(171M)，这是一个体积很小的词向量，方便快速加载，适合做`demo`使用。若想要使用更大规模的`GloVe`词向量，请到[官网](https://nlp.stanford.edu/projects/glove/)下载，官网中提到的`glove.6B.zip`(862M)中就包含了`glove.6B.{50d, 100d, 200d, 300d}.txt`四种规模。

#### 下载链接
为方便下载，我这里把`glove.6B.50d.txt`也放到了本仓库，下载后请**解压**后使用。

[glove.6B.50d.txt](https://github.com/skyhiter/torchtext-ignite-demo/tree/master/data)

### 实验环境
* macOS 10.14.5（macbook不支持CUDA加速）
* Python 3.6.8
* Pytorch 1.1.0
* text 0.4.0 [2019.05.22 master branch](`https://github.com/pytorch/text`)
* ignite 0.2.0 [2019.05.22 master branch](`https://github.com/pytorch/ignite`)

**注意**

如果通过`pip`安装`torchtext`与`ignite`：
* `pip install torchtext`
* `pip install pytorch-ignite`（这里是`pytorch-ignite`,不是`ignite`!!!）

目录文件说明：

```
./
├── README.md
├── data  // 语料，日志，缓存等路径; data/MR_10662.txt中的标签0是neg, 1是pos
├── data_loader.py // 使用torchtext预处理数据并加载
├── ignite    // ignite的源代码作为本地模块使用，自己修改了ignite/handlers/early_stopping.py
├── ignite_0.2.0_origin.zip // ignite的源代码v0.2.0仅备份用, 未做任何修改
├── log.py   // 全局logger，同时输出到文件和控制台
├── main.py  // 训练使用
├── myutils.py  // 常用函数
├── nnet  // 存放具体的模型实现
├── predict_local.py // 本地(区别于WebService)预测(推理)使用
└── torchtext   // torchtext的源代码作为本地模块使用，未作任何修改
```

### 实验结果
本实验为情感分析`demo`，仅使用CPU(`Intel i7-7920HQ@3.10GHz`,四核八线程)训练，每个`Epoch`约用时`15s`左右，未仔细调参数的情况下最终`acc`约为`0.7745`.
