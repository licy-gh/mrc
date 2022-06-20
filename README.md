# mrc_qa
中文机器阅读理解服务，包括生成式、抽取式和观点型三种模型，通过Flask与Swagger封装app。
## 环境
python >= 3.8：

```sh
pip install -r requirements.txt
```

**注：**[huggingface/transformers@`f5af873`](https://github.com/huggingface/transformers/commit/f5af87361718be29a1d3ddb2d8ef23f85b1c70c3) 修改了docstrings接口。

```diff
@add_code_sample_docstrings(
-    tokenizer_class=_TOKENIZER_FOR_DOC,
+    processor_class=_TOKENIZER_FOR_DOC,
     checkpoint=_CHECKPOINT_FOR_DOC,
     output_type=TFSequenceClassifierOutput,
     config_class=_CONFIG_FOR_DOC,
```

如果 `transformers >= 4.12` 会出现报错： `TypeError: add_code_sample_docstrings() got an unexpected keyword argument 'tokenizer_class'`。有两种解决方法：

1. 将transformers版本回滚至4.11之前
2. 将代码中的 `tokenizer_class` 重命名为 `processor_class`

## 运行

### 部署服务

```sh
python app.py
```

启动服务，同时在屏幕和 [MRC_integrated.log](./MRC_integrated.log) 记录运行日志，mrc_qa服务可通过两种方式访问（ip为部署服务的地址）：

1. url调用：采用post方式传入参数对 http://ip:8080/analysis/mrc_qa 进行调用，可使用 [postman](https://www.postman.com/) 进行测试。
2. 页面调用：http://ip:8080/apidocs 直接进行可视化页面调用。

### 接口格式

#### 输入参数

输入参数类型：接收`form-data`表单传入参数

| 字段名 | 类型   | 必填 | 描述                                                         |
| ------ | ------ | ---- | ------------------------------------------------------------ |
| query  | String | Y    | 用户问题                                                     |
| doc    | String | Y    | 文档                                                         |
| id     | String | Y    | 任务id，自定义                                               |
| flag   | String | N    | 使用哪种模型：gen生成模型，ext抽取模型，opn观点模型，为空默认抽取模型 |

示例：

```
id:testqa001
query:白云山有多高？
doc:白云山主峰海拔382m。
flag:gen
```

#### 输出参数

输出参数类型：json

| 字段名 | 类型   | 描述                                                         |
| ------ | ------ | ------------------------------------------------------------ |
| id     | string | 与任务id一致                                                 |
| code   | int    | 返回状态码。200：成功，500：接收参数时存在问题，501：接收成功，但算法执行过程中出现异常。 |
| msg    | string | 错误原因                                                     |
| data   | dict   | 模型结果。answer为回答，ans_start和ans_end为抽取式答案开始和结束的位置，生成和观点不使用为-1。 |
| time   | string | 模型消耗时间                                                 |

示例：

```json
{
    "code": 200,
    "data": {
        "ans_end": -1,
        "ans_start": -1,
        "answer": "382m"
    },
    "id": "t1",
    "msg": "成功",
    "time": 0.072
}
```

## 说明

```
mrc/
├── MRC_integrated.log	# mrc_qa服务运行日志
├── README.md			# mrc_qa说明文件
├── app.py				# mrc_qa封装代码
├── ext					# 抽取式模型
├── gen					# 生成式模型
├── opn					# 观点型模型
└── requirements.txt	# 包依赖文件
```

对于各个模型具体的训练和测试说明于对应目录下的README.md给出。

- [生成式说明](./gen/README.md)
- [抽取式说明](./ext/README.md)
- [观点型说明](./opn/README.md)
