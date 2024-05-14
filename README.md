# 蛋白质设计大赛

## 一、简介

### 单位：天津大学

### 作者：邱成浩 etc

### 目标

根据标注数据微调模型以适配任务——预测荧光蛋白强度，最终选择最佳荧光分数的蛋白质突变组合

### 代码参考

本项目基于 ProteinBERT 项目进行开发，ProteinBERT 源地址[ProteinBERT_Github] (https://github.com/nadavbra/protein_bert)
原项目 README.md 文件存储在`origin_README.md`中，以便于查看

### 环境配置

开发时使用 conda 环境已经导出，详见`environment.yml`

## 二、使用教程

### 1. 准备数据集

```bash
cd data
```

**在 data 文件夹中准备你需要的数据，分别为训练集和测试集**

在示例 data 中已经准备有相应的 python 脚本来辅助数据的处理

- `to_CSV.py` 将 excel 转化为 csv 文件
- `apply_mutations_CSV.py` 将突变信息转化为全序列信息
- `split.py` 以比例划分原始文件形成 train 和 test

在示例 data 中存储了两个版本的数据集（均进行了相应的划分）

- `GFP_data_with_full_sequences` 赛事主办方提供的数据，包含四种类型的荧光蛋白以及其突变信息与荧光分数
- `test` 取上述数据集的小样本，以便于测试代码各项功能

### 2. 进行微调

**微调代码储存在主目录下`FT.py`中**

- 在代码中可以调整训练、微调、最终轮数，不同的训练轮数和批量大小会显著影响训练的时间和质量
- 代码中增加了自定义回调函数，旨在每个 epoch 结束后测试其`MSE`与`MAE`作为性能指标
- 微调结束之后，整个模型（包括结构和权重）会保存在`./saved_model`路径

### 3. 加载模型并进行预测

**预测数据准备**

```bash
cd predict
```

`predict`文件夹中包含`predict_data`与 `predict_result`
其中`predict_data`包含需要测试的序列
`predict_result`为最终的结果输出

**预测代码储存在主目录下`predict.py`中**

- 在代码中可以调整进行预测的批量大小，这回影响预测的速度和消耗的资源
- 预测结束之后程序会在控制台打印相应的结果并且存储到`./predict/predict_result/predictions.csv`中

### 4. 筛选预测结果

**implementing**

## Citation <a name="citations"></a>
=======

If you use ProteinBERT, we ask that you cite our paper:
``` 
Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. & Linial, M. 
ProteinBERT: A universal deep-learning model of protein sequence and function. 
Bioinformatics (2022). https://doi.org/10.1093/bioinformatics/btac020
```

```bibtex
@article{10.1093/bioinformatics/btac020,
    author = {Brandes, Nadav and Ofer, Dan and Peleg, Yam and Rappoport, Nadav and Linial, Michal},
    title = "{ProteinBERT: a universal deep-learning model of protein sequence and function}",
    journal = {Bioinformatics},
    volume = {38},
    number = {8},
    pages = {2102-2110},
    year = {2022},
    month = {02},
    abstract = "{Self-supervised deep language modeling has shown unprecedented success across natural language tasks, and has recently been repurposed to biological sequences. However, existing models and pretraining methods are designed and optimized for text analysis. We introduce ProteinBERT, a deep language model specifically designed for proteins. Our pretraining scheme combines language modeling with a novel task of Gene Ontology (GO) annotation prediction. We introduce novel architectural elements that make the model highly efficient and flexible to long sequences. The architecture of ProteinBERT consists of both local and global representations, allowing end-to-end processing of these types of inputs and outputs. ProteinBERT obtains near state-of-the-art performance, and sometimes exceeds it, on multiple benchmarks covering diverse protein properties (including protein structure, post-translational modifications and biophysical attributes), despite using a far smaller and faster model than competing deep-learning methods. Overall, ProteinBERT provides an efficient framework for rapidly training protein predictors, even with limited labeled data.Code and pretrained model weights are available at https://github.com/nadavbra/protein\_bert.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac020},
    url = {https://doi.org/10.1093/bioinformatics/btac020},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/8/2102/45474534/btac020.pdf},
}
```