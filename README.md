## Pytorch torchtext and ignite demo

### 环境
* macOS 10.14.5
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
├── data  // 语料，缓存等路径
├── data_loader.py // 使用torchtext预处理数据并加载
├── ignite    // ignite的源代码作为本地模块使用，自己修改了ignite/handlers/early_stopping.py
├── ignite_0.2.0_origin.zip // ignite的源代码v0.2.0备份用, 未做任何修改
├── log.py   // 全局logger，同时输出到文件和控制台
├── main.py  // 训练使用
├── myutils.py
├── nnet  // 存放具体的模型实现
├── predict_local.py // 本地(区别于WebService)预测(推理)使用
└── torchtext   // torchtext的源代码作为本地模块使用，未作任何修改
```