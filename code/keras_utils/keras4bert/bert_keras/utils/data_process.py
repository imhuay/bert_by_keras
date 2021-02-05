#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-04 20:42
    
Author:
    huayang
    
Subject:
    数据处理工具函数

Notes:
    tf.Dataset 虽然支持多输入，但是不如单输入方便
"""

from multiprocessing.pool import ThreadPool

import numpy as np
import tensorflow as tf

from .backend import to_array, TF_FLOAT
from .tokenizer import tokenizer as _tokenizer

print(tf.__version__)


def _txt_to_array(txt1,
                  txt2=None,
                  label=None,
                  max_len=None,
                  one_hot=True,
                  tokenizer=_tokenizer):
    """"""
    token_ids, segment_ids = tokenizer.encode(txt1.numpy(), max_len=8)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # print(token_ids, segment_ids)
    return [token_ids, segment_ids]
    # txt_arr =
    # if label:
    #     return


def encode_fn(txt):
    """"""
    ret = tf.py_function(_txt_to_array, [txt], Tout=(tf.int64, tf.int64))

    return ret


def gen_data_set(data_path,
                 with_label=True,
                 with_txt2=False,
                 batch_size=32,
                 val_percent=0.,
                 is_shuffle=True,
                 label_mode='int',
                 n_class=None,
                 sep='\t',
                 max_len=120,
                 random_seed=1):
    """
    从文件生成训练集、验证集和测试集

    Notes:
        - Bert 是一个双输入模型（输出视情况），每个 step 的输入 shape 为 (2, batch_size, sequence_len)；
          为了构造这个形式的 data_set，需要一些特殊处理；
        - tf.Dataset 好像是支持直接生成多输入的，但是没在 API 里面找到示例，可以参考：
          使用`tf.data.Dataset`创建多输入Dataset-CSDN博客 | https://blog.csdn.net/qq_39238461/article/details/109160170）

    Args:
        data_path:
        with_label:
        with_txt2:
        batch_size:
        val_percent: 验证集比例，默认为 0.，即不划分验证集
        is_shuffle:
        label_mode: 标签类型，默认 'int'，传入其他值表示转成 one_hot 形式
        n_class: 默认为 None，若不传入，则以文件中 label 数量为准
        sep: 文件中每行的分隔符，默认 '\t'
        max_len: 序列长度，默认 120
        random_seed:

    Returns:

    """
    assert 0. <= val_percent < 1., 'val_percent 须在 [0, 1) 范围内。'
    is_val = val_percent > 0.

    def _encoder(txt1, txt2=None, label=None):
        tokens, segments = _tokenizer.encode(txt1, txt2, max_len)
        label = int(label) if label else label
        return tokens, segments, label

    def _get_ds(inp_token, inp_segment, inp_label):
        ds = tf.data.Dataset.from_tensor_slices((inp_token, inp_segment)).map(lambda x1, x2: [x1, x2])
        if with_label:
            ds_label = tf.data.Dataset.from_tensor_slices(inp_label)
            if label_mode != 'int':
                assert n_class is not None, 'label_mode != "int" 时，必须指定 n_class'
                ds_label = ds_label.map(lambda x: tf.one_hot(x, n_class, dtype=TF_FLOAT))
            ds = tf.data.Dataset.zip((ds, ds_label))
        ds = ds.batch(batch_size)
        return ds
    
    txt1_ls, txt2_ls, label_ls = [], [], []
    label_st = set()
    with open(data_path) as f:
        for ln in f:
            row = ln.strip().split(sep)
            txt1_ls.append(row[0])
            if len(row) <= 1:
                continue
                
            txt2_ls.append(row[1]) if with_txt2 else txt2_ls.append(None)
            label_ls.append(row[-1]) if with_label else label_ls.append(None)
            label_st.add(row[-1])
    
    if n_class is None:
        n_class = len(label_st)

    inp_token, inp_segment, inp_label = [], [], []
    with ThreadPool() as p:
        ret_iter = p.starmap(_encoder, zip(txt1_ls, txt2_ls, label_ls))
        for tokens, segments, label in ret_iter:
            inp_token.append(tokens)
            inp_segment.append(segments)
            inp_label.append(label)
    
    # shuffle 在划分验证集之前
    if is_shuffle:
        inp_zip = list(zip(inp_token, inp_segment, inp_label))
        rs = np.random.RandomState(random_seed)
        rs.shuffle(inp_zip)
        inp_token, inp_segment, inp_label = [list(it) for it in zip(*inp_zip)]  # 这里要把 tuple 转成 list

    inp_token_val, inp_segment_val, inp_label_val = [], [], []
    if is_val:
        n_val_samples = int(val_percent * len(inp_token))
        inp_token_val = inp_token[-n_val_samples:]
        inp_segment_val = inp_segment[-n_val_samples:]
        inp_label_val = inp_label[-n_val_samples:]
        
        inp_token = inp_token[:-n_val_samples]
        inp_segment = inp_segment[:-n_val_samples]
        inp_label = inp_label[:-n_val_samples]
        
    ds_train = _get_ds(inp_token, inp_segment, inp_label)
    
    if is_val:
        ds_val = _get_ds(inp_token_val, inp_segment_val, inp_label_val)
        return ds_train, ds_val
    
    return ds_train
