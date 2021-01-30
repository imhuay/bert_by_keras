#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-24 20:48
    
Author:
    huayang
    
Subject:
    extract_features

References:
    https://github.com/bojone/bert4keras/blob/master/examples/basic_extract_features.py
"""
try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from bert_keras.model import build_bret_from_config
from bert_keras.tokenizer import Tokenizer
from bert_keras.utils import to_array

config_path = '../model_ckpt/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../model_ckpt/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../model_ckpt/chinese_L-12_H-768_A-12/vocab.txt'

sequence_len = 100
assert sequence_len <= 512

# 加载模型
model_fix, model = build_bret_from_config(config_path, checkpoint_path,
                                          sequence_len=sequence_len,
                                          return_type='sentence_embedding', 
                                          return_full_model=True)
model.summary(line_length=200)

print('\n ===== Example of 获取 sentence embedding =====\n')
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

print('\n ----- Predicting -----\n')
ret = model_fix.predict(inputs)
print('outputs shape:', K.int_shape(ret))
print(ret)
"""
outputs shape: (2, 10, 768)
[[[-0.633944    0.20292063  0.08105    ...  0.49071276 -0.20267585
    0.25830606]
  [-0.75892895  0.09625201  1.0723153  ... -0.60993993  0.04389907
    0.03884725]
  [ 0.54979575 -0.7931236   0.4425914  ...  0.4254236   0.41041237
    0.0818297 ]
  ...
  [ 0.01977009 -0.36778107 -0.37989917 ...  0.6733829  -0.07508729
    0.06057744]
  [ 0.01823647 -0.34611243 -0.43257663 ...  0.6537404  -0.07068244
    0.07680589]
  [ 0.03579323 -0.4066236  -0.3740445  ...  0.6502873  -0.10985652
    0.04133184]]

 [[-0.13377029  0.11254838  0.13370925 ... -0.12091276 -0.46421608
    0.6820266 ]
  [ 0.01463162  0.22006218  0.42436084 ... -0.5977771  -0.74847466
    0.31168094]
  [ 0.72745985  0.02070418 -0.43255618 ... -0.12479839 -0.55839443
   -0.02365744]
  ...
  [ 0.01977    -0.36778125 -0.37989917 ...  0.6733826  -0.07508754
    0.06057742]
  [ 0.01823659 -0.34611243 -0.43257678 ...  0.6537404  -0.07068231
    0.07680591]
  [ 0.03579339 -0.40662354 -0.37404472 ...  0.6502874  -0.10985664
    0.04133194]]]

"""

# 测试中间层输出
# vector_funcrion = K.function([model.layers[0].input, model.layers[1].input], [model.get_layer('Transformer-0-MultiHeadSelfAttention').output])
# print(vector_funcrion([token_ids, segment_ids]))

print('\n ===== Example of 预测 Mask 单词 =====\n')
# 重构 model 的输出，相当于 return_type='mlm_probability'
model_fix = keras.Model(model.inputs, [model.outputs[2]], name='Bert-mlm')

text = u'数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科。'
print('待预测文本：', text)
print('Mask 的词：', '数学')
token_ids, segment_ids = tokenizer.encode(text, max_len=sequence_len)
token_ids[1] = token_ids[2] = tokenizer.mask_id
print('[MASK] 的 token id:', tokenizer.mask_id)
print('Mask 后的 token_ids:', token_ids)
print('Mask 后的 segment_ids:', segment_ids)
token_ids_inputs, segment_ids_inputs = to_array([token_ids], [segment_ids])
inputs = [token_ids_inputs, segment_ids_inputs]

print('\n ----- Predicting -----\n')
pred = model_fix.predict(inputs)
print(K.int_shape(pred))
pred_ids = pred[0][1:3].argmax(axis=1).tolist()
print('预测到的 token ids 及对应的字:', [(id_, tokenizer.inv_vocab[id_]) for id_ in pred_ids])  # [3144, 2110] -> ['数', '学']
"""
(1, 100, 21128)
预测到的 token ids 及对应的字: [(3144, '数'), (2110, '学')]
"""

print('\n ===== Example of 预测是否是下一个句子 =====\n')
# 重构 model 的输出，相当于 return_type='nsp_probability'
model_fix = keras.Model(model.inputs, [model.outputs[3]], name='Bert-nsp')

sentence = '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科。'
print('上一句：%s' % sentence)
sentence_1 = '从某种角度看属于形式科学的一种。'
print('待预测的下一句1：%s' % sentence_1)
sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'
print('待预测的下一句2：%s' % sentence_2)

token_ids_1, segment_ids_1 = tokenizer.encode(first=sentence, second=sentence_1, max_len=sequence_len)
print('第一组生成的 token_ids:', token_ids_1)
print('第一组生成的 segment_ids:', segment_ids_1)

token_ids_2, segment_ids_2 = tokenizer.encode(first=sentence, second=sentence_2, max_len=sequence_len)
print('第二组生成的 token_ids:', token_ids_2)
print('第二组生成的 segment_ids:', segment_ids_2)

token_ids_inputs = to_array([token_ids_1, token_ids_2])
segment_ids_inputs = to_array([segment_ids_1, segment_ids_2])
inputs = [token_ids_inputs, segment_ids_inputs]

print('\n ----- Predicting -----\n')
pred = model_fix.predict(inputs)
for i, it in enumerate(pred):
    print('第%s组是下一句的概率为：%.5f' % (i+1, it[0]))
"""
第1组是下一句的概率为：0.99991
第2组是下一句的概率为：0.00107
"""
