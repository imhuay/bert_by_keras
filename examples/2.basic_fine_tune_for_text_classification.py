#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-01 00:42
    
Author:
    huayang
    
Subject:
    使用 BERT fine tune 来进行文本分类任务
"""

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from bert_keras.model import build_bret_from_config


def build_model(config_path, checkpoint_path, sequence_len, n_class):
    """"""
    bert = build_bret_from_config(config_path, checkpoint_path,
                                  sequence_len=sequence_len,
                                  return_type='cls_embedding')  # 设置模型返回 cls_embedding
    # 获取 [CLS] 向量
    x = bert.output  # 这里用 output，outputs 返回的是一个列表
    x = keras.layers.Dropout(0.1, name='Classification_Dropout')(x)  # 如果用 outputs，这里要传 x[0]
    # 增加 softmax 层
    outputs = keras.layers.Dense(n_class, name='Classification_Softmax')(x)

    model = keras.Model(bert.inputs, outputs, name='Bert_Classification')
    return model


def main():
    """"""
    config_path = '../model_ckpt/chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = '../model_ckpt/chinese_L-12_H-768_A-12/bert_model.ckpt'
    vocab_path = '../model_ckpt/chinese_L-12_H-768_A-12/vocab.txt'

    n_epoch = 5
    sequence_len = 128
    n_class = 8

    model = build_model(config_path, checkpoint_path, sequence_len, n_class)
    model.summary(line_length=200)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(1e-5),  # 使用足够小的学习率，建议 1e-5 ~ 5e-5
        metrics=['accuracy'])

    # TODO: 准备数据
    # model.fit(ds_train, epochs=n_epoch, verbose=1)


if __name__ == '__main__':
    """"""
    main()
