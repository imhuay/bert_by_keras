#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-01 00:42

Author:
    huayang

Subject:
    如何 fine tune BERT 来进行文本分类任务

运行环境:
    - tensorflow==2.4
    - tensorflow==2.2
"""
import os
import argparse

# import tensorflow_addons as tfa

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from bert_by_keras.model.bert import build_bret, bert_output_adjust, model_fine_tune_config
from bert_by_keras.utils.data_process import get_data_set_basic
# from bert_keras.optimizer.weight_decay_optimizers import AdamW


def build_model(args):
    """"""
    config_path = os.path.join(args.ckpt_path, 'bert_config.json')
    checkpoint_path = os.path.join(args.ckpt_path, 'bert_model.ckpt')

    bert, layer_name = build_bret(config_path, checkpoint_path,
                                  sequence_len=args.sequence_len,
                                  return_layer_name=True)  # 设置模型返回 cls_embedding

    bert = bert_output_adjust(bert, output_type='cls_embedding')
    # 获取 [CLS] 向量
    x = bert.output  # 这里用 output，outputs 返回的是一个列表
    x = keras.layers.Dropout(0.1, name='Classification_Dropout')(x)  # 如果用 outputs，这里要传 x[0]
    # 增加 softmax 层
    outputs = keras.layers.Dense(args.n_class, activation='softmax', name='Classification_Softmax')(x)

    model = keras.Model(bert.inputs, outputs, name='Bert_Classification')

    # AdamW 效果很差，可能原因：没有配合 lr 衰减
    # optimizer = AdamW(weight_decay=0.01, learning_rate=2e-5, epsilon=1e-6)
    optimizer = keras.optimizers.Adam(2e-5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,  # 使用足够小的学习率，建议 1e-5 ~ 5e-5
                  metrics=['accuracy'])

    return model, layer_name


def get_ds(args):
    """"""
    # 数据准备
    val_percent = 0. if args.file_path_val else 0.2
    ds = get_data_set_basic(args.file_path_train,
                            n_class=args.n_class,
                            max_len=args.sequence_len,
                            batch_size=args.batch_size,
                            with_label=args.with_label,
                            with_txt2=args.with_txt2,
                            label_mode=args.label_mode,
                            val_percent=val_percent)

    if args.file_path_val:
        ds_train = ds
        ds_val = get_data_set_basic(args.file_path_val,
                                    n_class=args.n_class,
                                    max_len=args.sequence_len,
                                    batch_size=args.batch_size,
                                    with_label=args.with_label,
                                    with_txt2=args.with_txt2,
                                    label_mode=args.label_mode)
    else:
        ds_train, ds_val = ds

    return ds_train, ds_val


def fine_tune(args):
    """"""
    model, layer_name = build_model(args)

    # 加载已经训练的权重
    if args.save_path:
        print('\n--- 加载模型权重 ---\n')
        model.load_weights(args.save_path)

    ds_train, ds_val = get_ds(args)

    model.fit(ds_train, epochs=args.n_epoch, validation_data=ds_val, verbose=1)

    if args.save_path:
        # model.save(args.save_path)
        model.save_weights(args.save_path)


def get_args():
    """"""
    p = argparse.ArgumentParser(description='')

    p.add_argument('--ckpt_path',  #
                   type=str,
                   default=r'../model_file/chinese_wwm_ext_L-12_H-768_A-12',
                   help='预训练模型', )

    p.add_argument('--file_path_train',  #
                   default=r'../data_set/test_data.txt',
                   type=str,
                   help='训练集路径', )

    p.add_argument('--file_path_val',  #
                   type=str,
                   help='验证集路径', )

    p.add_argument('--save_path',  #
                   type=str,
                   default='./_out/ckpt/model',
                   help='模型保存路径', )

    p.add_argument('--batch_size',  #
                   default=8,
                   type=int, )

    p.add_argument('--sequence_len',  #
                   default=128,
                   type=int, )

    p.add_argument('--n_epoch',  #
                   default=5,
                   type=int, )

    p.add_argument('--n_class',  #
                   default=2,
                   type=int, )

    p.add_argument('--with_label',  #
                   action='store_false', )

    p.add_argument('--with_txt2',  #
                   action='store_true', )

    p.add_argument('--label_mode',  #
                   default='one_hot',
                   type=str, )

    return p.parse_args()


if __name__ == '__main__':
    """"""
    args = get_args()

    vocab_path = os.path.join(args.ckpt_path, 'vocab.txt')

    model, layer_name = build_model(args)

    # 打印模型结构
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

    # 训练集、验证集
    ds_train, ds_val = get_ds(args)

    # 训练并保存模型
    print('\n===== 开始训练 =====\n')
    model.fit(ds_train, epochs=args.n_epoch, validation_data=ds_val, verbose=1)

    print('\n===== 保存模型 =====\n')
    if args.save_path:
        # model.save(args.save_path)
        model.save_weights(args.save_path)

    print('\n===== 模型 fine tune =====\n')
    fine_tune(args)
