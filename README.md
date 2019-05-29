## Pytorch torchtext and ignite demo

### 实验语料
本demo使用的是经典的MR(Movie Review)情感分析语料，共10662条句子。官网为http://www.cs.cornell.edu/people/pabo/movie-review-data , 需下载其中的`sentence polarity dataset v1.0`。

#### 下载链接
(Movie Review Data)[http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz]

### 实验环境
* macOS 10.14.5（不支持CUDA加速）
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
├── data  // 语料，日志，缓存等路径
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
本实验为情感分析demo，仅使用CPU(`Intel i7-7920HQ@3.10GHz`,四核八线程)训练，每个`Epoch`约用时`15s`左右，未仔细调参数的情况下最终`acc`约为`0.7745`.
