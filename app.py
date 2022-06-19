# -*- coding: utf-8 -*-
import logging
import os
import sys

sys.path.append('./gen')
sys.path.append('./ext')
sys.path.append('./opn')
from flasgger import Swagger
from flask import Flask, request
import time
import torch
from gen.rc import GenAnswer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from opn.main import Config as OpnConfig
from opn.main import Model as OpnModel
from opn.main import prediction_model as opn_answer

app = Flask(__name__)  # Flask web服务包装
Swagger(app)  # swagger页面包装
curr_path = os.path.dirname(__file__)
# 日志设置，同时输出到终端和文件
logging.root.handlers = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(
            filename=os.path.join(curr_path, "MRC_integrated.log"),
            mode="w",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Chinese MRC")
# gen
gen = GenAnswer()

# ext
ext_qa = pipeline(
    task='question-answering',
    model=AutoModelForQuestionAnswering.from_pretrained(
        os.path.join(curr_path, "./ext/Output/model")
    ),
    tokenizer=AutoTokenizer.from_pretrained(
        os.path.join(curr_path, "./ext/Output/model")
    )
)

# opn
opn_config = OpnConfig()
opn_model = OpnModel(opn_config).to(opn_config.device)
opn_model.load_state_dict(torch.load(opn_config.save_path, map_location='cpu'))


@app.route('/analysis/mrc_qa', methods=['POST'])  # web服务url定义
def main():
    # ################### 该部分是生成API接口文档的解析部分 start，仿照下面调整###############
    """
    中文机器阅读理解服务的API接口
    输入篇章和问题，进行阅读理解返回答案
    ---
    tags:
      - 中文机器阅读理解 API， 支持根据用户输入的文档和问题回答答案
    parameters:
      - name: id
        in: formData
        type: string
        required: true
        description: 任务id
      - name: query
        in: formData
        type: string
        required: true
        description: 用户问题
      - name: doc
        in: formData
        type: string
        required: true
        description: 文档
      - name: flag
        in: formData
        type: string
        required: false
        description: 使用哪种模型, "gen"表示生成模型, "ext"表示抽取模型, "opn"表示观点问题, 为空则默认抽取模型
    responses:
      500:
        description: Error：接收参数时存在问题!
      200:
        description: 成功，阅读理解的回答
        schema:
          id: results
          properties:
            id:
              type: string
              description: 与任务id相一致
              default:
            code:
              type: integer
              description: 状态编号，200是成功，500是接收参数时存在问题，501是接收成功，但算法执行过程中出现异常
              default:
            msg:
              type: string
              description: 报错信息
              default:
            data:
              type: array
              description: 模型结果
              default:
            time:
              type: number
              description: 模型耗时
              default:
    """
    # ##################### 生成API文档描述部分 END  ##########################

    # 接收form-data表单传入参数
    taskID = request.form.get('id')
    query = request.form.get('query')
    doc = request.form.get('doc')
    flag = request.form.get('flag', default='ext')
    flag = flag if flag else 'ext'
    # 接收body-raw的json传入参数
    # content = request.get_data()
    receive_info = {
        "id": taskID,
        "code": 500,
        "msg": None,
        "data": {
            "answer": "",
            "ans_start": -1,
            "ans_end": -1
        },
        "time": 0.0
    }
    if query and doc and flag in ['gen', 'ext', "opn"]:
        try:
            if flag == 'gen':
                # 生成
                t0 = time.time()
                ans = gen.answer(query, doc)
                t1 = time.time()
                elapsed = round(t1 - t0, 4)
                receive_info["data"]["answer"] = ans
                receive_info["time"] = elapsed
            elif flag == 'ext':
                # 抽取
                t0 = time.time()
                ans_pack = ext_qa({
                    "question": query,
                    "context": doc
                })
                t1 = time.time()
                elapsed = round(t1 - t0, 4)
                receive_info["data"]["answer"] = ans_pack["answer"]
                receive_info["data"]["ans_start"] = ans_pack["start"]
                receive_info["data"]["ans_end"] = ans_pack["end"]
                receive_info["time"] = elapsed
            elif flag == 'opn':
                # 观点
                t0 = time.time()
                ans = opn_answer(query + doc)
                t1 = time.time()
                elapsed = round(t1 - t0, 4)
                receive_info["data"]["answer"] = ans
                receive_info["time"] = elapsed
            else:
                pass
            receive_info["code"] = 200
            receive_info["msg"] = "成功"

        except Exception as err:
            logging.error(err)
            logger.error(f"taskID: {taskID} query: {query} doc: {doc} flag: {flag}")
            logger.error(f"receive_info: {receive_info}")
            receive_info["code"] = 501
            receive_info["msg"] = "接收成功，但算法执行过程中出现异常"
    else:
        receive_info["code"] = 500
        receive_info["msg"] = "接收参数时存在问题"
    return receive_info


if __name__ == '__main__':
    # 启动服务，可通过两种方式访问：
    # （1）url调用：采用post方式传入参数对http://ip:8080/analysis/mrc_qa 进行调用
    # （2）页面调用：http://ip:8080/apidocs  进行可视化页面调用
    app.run(host='0.0.0.0', port=8080)
