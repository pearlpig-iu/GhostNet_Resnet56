# GhostNet_Resnet56

# 依赖

Python 3.0+  
PyTorch 1.0+  
tensorboard 2.0.0  
torchstat 0.0.7    

# 数据准备

首先准备好数据集，创建data文件夹，将数据集配置为如下的文件结构：（类别为文件名）

```
data
├── train
│   ├──	class1
│   │   ├── 026.JPEG
│   │   ├── ...
│   ├── class2
│   │   ├── 999.JPEG
│   │   ├── ...
│   ├── ...
├── val
│   ├── class1
│   │   ├── 0027.JPEG
│   │   ├── ...
│   ├── class2
│   │   ├── 993.JPEG
│   │   ├── ...
│   ├── ...
├── test
│   ├── class1
│   │   ├── 0067.JPEG
│   │   ├── ...
│   ├── class2
│   │   ├── 8983.JPEG
│   │   ├── ...
│   ├── ...
```

# 模型训练

运行trian.py即可

# 模型测试及推理

运行test.py进行测试，test文件需按上述格式配置

运行inference.py对单张图片进行推理

## 参数计算

执行 compute\_flops.py  
需要安装torchstat  
安装方法： pip install torchstat  
torchstat网站：https://github.com/Swall0w/torchstat. 
