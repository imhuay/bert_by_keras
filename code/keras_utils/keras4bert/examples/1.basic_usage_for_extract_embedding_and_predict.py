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
"""
try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from bert_keras.model import build_bret
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
print(ret)
"""
outputs shape: (2, 100, 768)
[[[-0.6325102   0.20302321  0.07936531 ...  0.49122596 -0.20493448
    0.2575256 ]
  [-0.7588352   0.09651878  1.0718747  ... -0.61096907  0.04312116
    0.03881412]
  [ 0.5477028  -0.79211754  0.4443523  ...  0.42449158  0.41105708
    0.08222917]
  ...
  [-0.03277081 -0.48243824 -0.361671   ...  0.7180104  -0.12803611
    0.03498956]
  [-0.02219212 -0.48458925 -0.43217662 ...  0.7270943  -0.07406927
    0.01058598]
  [-0.06935356 -0.49976897 -0.3491079  ...  0.7284312  -0.1276668
    0.03296683]]

 [[-0.1333287   0.11290775  0.13338596 ... -0.12072507 -0.46391013
    0.682771  ]
  [ 0.01600987  0.2203105   0.42461488 ... -0.5974555  -0.7474879
    0.3116304 ]
  [ 0.72724533  0.01966706 -0.4315258  ... -0.1256972  -0.5581709
   -0.02299952]
  ...
  [-0.03277081 -0.48243824 -0.361671   ...  0.7180104  -0.12803611
    0.03498956]
  [-0.02219215 -0.48458892 -0.43217683 ...  0.7270941  -0.07406903
    0.010586  ]
  [-0.06935389 -0.49976864 -0.34910777 ...  0.7284312  -0.12766662
    0.03296699]]]
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
pred_ids = pred[0][1:3].argmax(axis=1).tolist()
print('预测的 token_id 及对应的字:', [(id_, tokenizer.inv_vocab[id_]) for id_ in pred_ids])  # [3144, 2110] -> ['数', '学']
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
pred = model_fix.predict(inputs)
for i, it in enumerate(pred):
    print('第%s组是下一句的概率为：%.5f' % (i+1, it[0]))
"""
第1组是下一句的概率为：0.99991
第2组是下一句的概率为：0.00109
"""
