#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-01 00:42
    
Author:
    huayang
    
Subject:
    如何 fine tune BERT 来进行文本分类任务
"""

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from bert_keras.model.bert import build_bret, model_fine_tune_config
from bert_keras.utils.data_process import gen_data_set


def build_model(config_path, checkpoint_path, sequence_len, n_class):
    """"""
    bert, layer_name = build_bret(config_path, checkpoint_path,
                                  sequence_len=sequence_len,
                                  output_type='cls_embedding',
                                  return_layer_name=True)  # 设置模型返回 cls_embedding
    # 获取 [CLS] 向量
    x = bert.output  # 这里用 output，outputs 返回的是一个列表
    x = keras.layers.Dropout(0.1, name='Classification_Dropout')(x)  # 如果用 outputs，这里要传 x[0]
    # 增加 softmax 层
    outputs = keras.layers.Dense(n_class, name='Classification_Softmax')(x)

    model = keras.Model(bert.inputs, outputs, name='Bert_Classification')
    return model, layer_name


def main():
    """"""
    model_name = 'chinese_wwm_ext_L-12_H-768_A-12'
    config_path = '../model_ckpt/%s/bert_config.json' % model_name
    checkpoint_path = '../model_ckpt/%s/bert_model.ckpt' % model_name
    vocab_path = '../model_ckpt/%s/vocab.txt' % model_name

    n_epoch = 5
    sequence_len = 128
    n_class = 2

    model, layer_name = build_model(config_path, checkpoint_path, sequence_len, n_class)
    model.summary(line_length=200)

    # 控制微调哪些层
    model = model_fine_tune_config(
        model, layer_name,
        is_fine_tune=True,
        is_fine_tune_norm=False,  # layer norm 不参与微调
        is_fine_tune_embedding=False,  # embedding 不参与微调
        n_fine_tune_transformer_block=3  # 只微调 3 个 transformer_block（接近输出层的 block）
    )

    # 打印参与微调的层
    print('\n===== 参与 fine tune 的层 =====\n')
    for layer in model.layers:
        if layer.trainable_weights:  # 使用 trainable 判断会把所有层都打印出来，不知道为什么
            print(layer.name)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(2e-5),  # 使用足够小的学习率，建议 1e-5 ~ 5e-5
                  metrics=['accuracy'])

    # 数据准备
    data_path = r'../data_set/lcqmc_demo/lcqmc.train.data'
    ds_train, ds_val = gen_data_set(data_path,
                                    max_len=sequence_len,
                                    batch_size=8,
                                    # with_label=False,
                                    val_percent=0.2)

    # 训练
    for x, y in ds_train.take(1):
        print(K.shape(x))
        print(K.shape(y))
        # print(it)
    model.fit(ds_train, epochs=n_epoch, validation_data=ds_val, verbose=1)


if __name__ == '__main__':
    """"""
    main()
