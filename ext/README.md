# mrc_qa_ext

## 环境

同 [mrc_qa](../README.md) 的环境配置。

## 数据集

融合了WebQA、ChineseSquad、Sogou QA、军事QA、Dureader、CMRC2018、法研杯CAIL、疫情问答、DRCD，经过预处理为统一格式，以9：1划分训练集和验证/测试集。

## 预训练模型

基于MRC数据再训练的开源模型：roberta-wwm-ext-large（https://github.com/luhua-rain/MRC_Competition_Dureader）

## 训练

```python
python run.py
--max_len=512
--model_name_or_path= #预训练模型路径
--per_gpu_train_batch_size=7
--per_gpu_eval_batch_size=40
--learning_rate=1e-5
--linear_learning_rate=1e-4
--num_train_epochs=100
--output_dir= #输出路径
--weight_decay=0.01
--early_stop=2
```

## 预测

```PYTHON
python predict.py
--max_len=400
--model_name_or_path= #预训练模型路径
--per_gpu_eval_batch_size=120
--output_dir= #输出路径
--fine_tunning_model= #微调后模型路径
```

## 结果

使用F1-score和EM-score作为评估方式：

| F1-score           | EM-score           |
| ------------------ | ------------------ |
| 0.7150976984080366 | 0.6171681415929204 |

## 说明

```
ext
│  main.ipynb	#使用jupyter运行代码
│  main.py		#封装
│  predict.py	#预测
│  README.md	#说明文件
│  run.py		#训练模型
├─data			#训练数据
├─log			#训练日志
├─luhua			#预训练模型
├─metric		#保存评估文件
├─models		#保存模型文件
├─mydataset		#数据处理
├─Output		#保留输出结果
├─utils			#辅助函数
└─__pycache__
```

