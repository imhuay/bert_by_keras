#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-05 01:14
    
Author:
    huayang
    
Subject:
    
"""

from bert_keras.utils.data_process import gen_data_set
import keras.backend as K

if __name__ == '__main__':
    """"""
    ds, ds_val = gen_data_set(r'../data_set/lcqmc_demo/lcqmc.train.data',
                              with_label=True,
                              with_txt2=True,
                              is_shuffle=True,
                              val_percent=0.4,
                              label_mode='one_hot',
                              batch_size=3,
                              max_len=10)

    for it in ds.take(1):
        # print(K.shape(it[0]), K.shape(it[1]))
        print(it[0])
        print(it[1])

    print()
    for it in ds_val:
        print(K.shape(it[0]), K.shape(it[1]))
    
    """
    tf.Tensor([  2   3 120], shape=(3,), dtype=int32) tf.Tensor([3 2], shape=(2,), dtype=int32)
    tf.Tensor([  2   3 120], shape=(3,), dtype=int32) tf.Tensor([3 2], shape=(2,), dtype=int32)
    
    tf.Tensor([  2   3 120], shape=(3,), dtype=int32) tf.Tensor([3 2], shape=(2,), dtype=int32)
    tf.Tensor([  2   1 120], shape=(3,), dtype=int32) tf.Tensor([1 2], shape=(2,), dtype=int32)
    """
