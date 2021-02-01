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
                              output_type='sentence_embedding',
                              return_full_model=True,
                              return_config=False)
model.summary(line_length=200)

print('\n===== 1. Example of 获取 sentence embedding =====')
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
[[[-0.6325102   0.20302321  0.07936531 -0.03284243  0.5668083 ]]

 [[-0.1333287   0.11290775  0.13338596  0.40458745  0.4347366 ]]]
"""

# 测试中间层输出
# vector_funcrion = K.function(model.inputs, model.get_layer('Transformer-0-MultiHeadSelfAttention').output)
# print(vector_funcrion(inputs))

print('\n===== 2. Example of 预测 Mask 单词 =====')
# 重构 model 的输出，这个例子中相当于 return_type='mlm_probability'
model_fix = keras.Model(model.inputs, model.outputs[2], name='Bert-mlm')

text = u'数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科。'
print('待预测文本："%s"' % text)
print('其中被 Mask 的词："%s"' % '数学')
token_ids, segment_ids = tokenizer.encode(text, max_len=sequence_len)
print('把"数"和"学"的 token_id 替换成 "[MASK]" 的 token_id。')
token_ids[1] = token_ids[2] = tokenizer.mask_id
print('其中 "[MASK]" 的 token_id 为:', tokenizer.mask_id)
print('文本被 Mask 后的 token_ids:', token_ids)
print('文本被 Mask 后的 segment_ids:', segment_ids)
token_ids_inputs, segment_ids_inputs = to_array([token_ids], [segment_ids])
inputs = [token_ids_inputs, segment_ids_inputs]

print('\n--- Predicting ---')
pred = model_fix.predict(inputs)
print(K.int_shape(pred))
pred_ids = pred[0][1:3].argmax(axis=1).tolist()  # 提取预测结果的 token id
# 期望结果
ret_except = [(3144, '数'), (2110, '学')]
# 实际结果
ret = [(id_, tokenizer.inv_vocab[id_]) for id_ in pred_ids]
print('预测的 token_id 及对应的字:', ret)  # [3144, 2110] -> ['数', '学']
assert ret == ret_except, '实际结果与期望值不符'
"""
(1, 100, 21128)
预测的 token_id 及对应的字: [(3144, '数'), (2110, '学')]
"""

print('\n===== 3. Example of 预测是否是下一个句子 =====')
# 重构 model 的输出，这个例子中相当于 return_type='nsp_probability'
model_fix = keras.Model(model.inputs, model.outputs[3], name='Bert-nsp')

sentence = '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科。'
print('上一句："%s"' % sentence)
sentence_1 = '从某种角度看属于形式科学的一种。'
print('待预测的下一句1："%s"' % sentence_1)
sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'
print('待预测的下一句2："%s"' % sentence_2)

token_ids_1, segment_ids_1 = tokenizer.encode(first=sentence, second=sentence_1, max_len=sequence_len)
print('第一组生成的 token_ids:', token_ids_1)
print('第一组生成的 segment_ids:', segment_ids_1)

token_ids_2, segment_ids_2 = tokenizer.encode(first=sentence, second=sentence_2, max_len=sequence_len)
print('第二组生成的 token_ids:', token_ids_2)
print('第二组生成的 segment_ids:', segment_ids_2)

token_ids_inputs = to_array([token_ids_1, token_ids_2])
segment_ids_inputs = to_array([segment_ids_1, segment_ids_2])
inputs = [token_ids_inputs, segment_ids_inputs]

print('\n--- Predicting ---')
ret = model_fix.predict(inputs)
# print(ret.tolist())
# 期望结果
ret_except = np.array([[0.9999082088470459, 9.180504275718704e-05], [0.0010862667113542557, 0.9989137649536133]])
assert np.allclose(ret, ret_except, atol=0.001), '实际结果与期望值不符'
for i, it in enumerate(ret):
    print('第%s组是下一句的概率为：%.5f' % (i + 1, it[0]))
"""
第1组是下一句的概率为：0.99991
第2组是下一句的概率为：0.00109
"""
