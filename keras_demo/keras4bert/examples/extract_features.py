#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-24 20:48
    
Author:
    huayang
    
Subject:
    
"""
try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

# from bert4keras.tokenizers import Tokenizer
from bert_keras.model import build_bret_from_config
from bert_keras.tokenizer import Tokenizer
from bert_keras.utils import to_array

config_path = '../model_ckpt/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../model_ckpt/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../model_ckpt/chinese_L-12_H-768_A-12/vocab.txt'

model = build_bret_from_config(config_path, checkpoint_path)
# model = build_bret_from_config('bert_keras/bert_config.json', checkpoint_path)
model.summary(line_length=200)

# tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
tokenizer = Tokenizer(dict_path)  # 建立分词器
token_ids, segment_ids = tokenizer.encode(u'语言模型')
print(token_ids)
print(segment_ids)
# token_ids, segment_ids = tokenizer.encode(u'语言模型')
token_ids, segment_ids = to_array([token_ids, token_ids], [segment_ids, segment_ids])
print(K.int_shape(token_ids), K.int_shape(segment_ids))

print('\n ===== predicting =====\n')
ret = model.predict([token_ids, segment_ids])
print(K.int_shape(ret))
print(ret)
"""
(2, 6, 768)
[[[-0.63250995  0.20302348  0.07936601 ...  0.4912259  -0.20493391
    0.25752544]
  [-0.7588355   0.09651889  1.0718753  ... -0.61096895  0.04312206
    0.03881405]
  [ 0.547703   -0.7921168   0.44435176 ...  0.42449194  0.41105705
    0.08222881]
  [-0.29242465  0.6052718   0.4996871  ...  0.86041427 -0.65331763
    0.5369073 ]
  [-0.7473455   0.49431536  0.71851677 ...  0.38486052 -0.74090594
    0.3905684 ]
  [-0.87413853 -0.21650384  1.3388393  ...  0.5816851  -0.43732336
    0.5618182 ]]

 [[-0.6325102   0.20302393  0.07936587 ...  0.49122596 -0.20493367
    0.25752553]
  [-0.7588355   0.09651913  1.0718752  ... -0.61096907  0.04312139
    0.03881442]
  [ 0.54770297 -0.7921169   0.44435278 ...  0.42449135  0.41105768
    0.08222892]
  [-0.29242444  0.60527134  0.4996862  ...  0.8604133  -0.6533168
    0.5369075 ]
  [-0.7473458   0.49431637  0.7185164  ...  0.38486147 -0.7409059
    0.3905691 ]
  [-0.87413836 -0.21650326  1.3388393  ...  0.5816858  -0.437323
    0.5618183 ]]]
"""

# 测试中间层输出
# vector_funcrion = K.function([model.layers[0].input, model.layers[1].input], [model.get_layer('Transformer-0-MultiHeadSelfAttention').output])
# print(vector_funcrion([token_ids, segment_ids]))
