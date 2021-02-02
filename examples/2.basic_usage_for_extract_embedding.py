#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-24 20:48
    
Author:
    huayang
    
Subject:
    extract_features
    - 不同版本的 tf 版本输出可能会略有区别
    - 以下输出基于 tf2.4 版本

References:
    https://github.com/bojone/bert4keras/blob/master/examples/basic_extract_features.py
    https://github.com/CyberZHG/keras-bert/blob/master/demo/load_model/load_and_predict.py

Other:
    keras<2.4 或 tensorflow<2.4 的版本，tf.keras 和 keras 不能混用，之后的版本应该可以（未测试）
    - 比如 keras==2.3.1 下 Input 层的 output_shape 为 (None, sequence_len)，而 keras==2.4.3 中为 [(None, sequence_len)]
"""
import numpy as np

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from bert_keras.model.bert import build_bret
from bert_keras.tokenizer import Tokenizer
from bert_keras.utils import to_array

config_path = '../model_ckpt/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../model_ckpt/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../model_ckpt/chinese_L-12_H-768_A-12/vocab.txt'

sequence_len = 100
assert sequence_len <= 512

# 加载模型：这里 model 为完整模型，包含多个输出；model_fix 为根据 return_type 裁剪输出后的模型
model_fix, model = build_bret(config_path, checkpoint_path,
                              sequence_len=sequence_len,
                              output_type='sentence_embedding',  # 输出为 sentence embedding
                              return_full_model=True,  # 返回完整模型，用于第二个例子中重构输出
                              return_config=False)
model.summary(line_length=200)

print('\n===== 1. Example of extract sentence embedding =====')
tokenizer = Tokenizer(dict_path)  # 建立分词器
token_ids_1, segment_ids_1 = tokenizer.encode(u'语言模型', max_len=sequence_len)
print('token_ids of "[CLS]语言模型[SEP]":', token_ids_1)
print('segment_ids of "[CLS]语言模型[SEP]":', segment_ids_1)
token_ids_2, segment_ids_2 = tokenizer.encode(u'深度学习', max_len=sequence_len)
print('token_ids of "[CLS]深度学习[SEP]":', token_ids_2)
print('segment_ids of "[CLS]深度学习[SEP]":', segment_ids_2)

# Batch token_ids to input array
token_ids_inputs = to_array([token_ids_1, token_ids_2])
segment_ids_inputs = to_array([segment_ids_1, segment_ids_2])
print('token_ids_inputs shape:', K.int_shape(token_ids_inputs))
print('segment_ids_inputs shape:', K.int_shape(segment_ids_inputs))
inputs = [token_ids_inputs, segment_ids_inputs]

print('\n--- Predicting ---')
ret = model_fix.predict(inputs)
print('outputs shape:', K.int_shape(ret))
# 截断结果
ret_part = ret[:, :1, :5]
# print(ret_part.tolist())
# 期望结果（截断）
ret_part_except = np.array(
    [[[-0.6325101852416992, 0.20302321016788483, 0.0793653056025505, -0.032842427492141724, 0.56680828332901]],
     [[-0.13332870602607727, 0.11290775239467621, 0.13338595628738403, 0.40458744764328003, 0.43473660945892334]]]
)
print('部分截断结果：')
print(ret_part)
assert np.allclose(ret_part, ret_part_except, atol=0.003), '实际结果与期望值不符'
"""
outputs shape: (2, 100, 768)
部分截断结果：
[[[-0.6325102   0.20302321  0.07936531 -0.03284243  0.5668083 ]]

 [[-0.1333287   0.11290775  0.13338596  0.40458745  0.4347366 ]]]
"""

# 测试中间层输出
# vector_funcrion = K.function(model.inputs, model.get_layer('Transformer-0-MultiHeadSelfAttention').output)
# print(vector_funcrion(inputs))

print('\n===== 2. Example of extract cls embedding =====')
# 重构 model 的输出，这个例子中相当于 return_type='cls_embedding'
model_fix = keras.Model(model.inputs, model.outputs[1], name='Bert-cls')

print('\n--- Predicting ---')
ret = model_fix.predict(inputs)
print('outputs shape:', K.int_shape(ret))
# 截断结果
ret_part = ret[:, :5]
# print(ret_part.tolist())
# 期望结果（截断）
ret_part_except = np.array(
    [[0.994243860244751, 0.9995892643928528, 0.9286629557609558, 0.4422328770160675, 0.9044715166091919],
     [0.9903159737586975, 0.9998862743377686, 0.8060687780380249, 0.6435789465904236, 0.878944993019104]]
)
print('部分截断结果：')
print(ret_part)
assert np.allclose(ret_part, ret_part_except, atol=0.003), '实际结果与期望值不符'
"""
outputs shape: (2, 768)
部分截断结果：
[[0.99424386 0.99958926 0.92866296 0.44223288 0.9044715 ]
 [0.990316   0.9998863  0.8060688  0.64357895 0.878945  ]]
"""
