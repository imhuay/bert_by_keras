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

config_path = '/Users/huayang/workspace/model/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/huayang/workspace/model/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/huayang/workspace/model/bert/chinese_L-12_H-768_A-12/vocab.txt'

model = build_bret_from_config('bert_keras/bert_config.json', checkpoint_path, sequence_len=512)
# model = build_bret_from_config('bert_keras/bert_config.json', checkpoint_path)
model.summary(line_length=200)

# tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
tokenizer = Tokenizer(dict_path)  # 建立分词器
token_ids, segment_ids = tokenizer.encode(u'语言模型', max_len=512)
print(token_ids)
print(segment_ids)
# token_ids, segment_ids = tokenizer.encode(u'语言模型')
token_ids, segment_ids = to_array([token_ids, token_ids], [segment_ids, segment_ids])
print(K.int_shape(token_ids), K.int_shape(segment_ids))

print('\n ===== predicting =====\n')
ret = model.predict([token_ids, segment_ids])
print(K.int_shape(ret))
print(ret)
print(ret[0][0][0])

# assert float(ret[0][0][0]) == -0.6325102
# assert float(ret[0][1][0]) == -0.758836
# assert float(ret[0][2][0]) == 0.547703

# 中间层输出
# vector_funcrion = K.function([model.layers[0].input, model.layers[1].input], [model.get_layer('Transformer-0-MultiHeadSelfAttention').output])
# print(vector_funcrion([token_ids, segment_ids]))