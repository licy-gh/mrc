# mrc_qa_opt

## 环境

同 [mrc_qa](../README.md) 的环境配置。

## 数据集

使用dureader yesno数据集，经过预处理为统一格式。训练集数据71789条，测试集数据3602条。

## 预训练模型

bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
	

词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
		

来自https://github.com/huggingface/pytorch-transformers

## 训练、验证、测试

```python
python run.py --model bert
```

## 结果

```
              precision    recall  f1-score   support

         Yes     0.8198    0.9200    0.8670      1874
          No     0.8618    0.7698    0.8132      1199
     Depends     0.7103    0.5747    0.6353       529

    accuracy                         0.8193      3602
   macro avg     0.7973    0.7548    0.7718      3602
weighted avg     0.8177    0.8193    0.8151      3602
```

## 说明

```
opt
│  LICENSE
│  main.py				#封装
│  README.md			#说明文件
│  run.py				#模型训练
│  train_eval.py		#模型评估
│  Untitled.ipynb		#使用jupyter运行代码
│  utils.py				#辅助函数
├─.ipynb_checkpoints
├─bert_pretrain			#bert预训练模型
├─ERNIE_pretrain		#ERNIE预训练模型
├─models				#模型与参数
├─pytorch_pretrained	
└─THUCNews				#dureader数据集，格式与THUCNews相同
```

