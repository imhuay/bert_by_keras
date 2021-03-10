#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-24 18:17
    
Author:
    huayang
    
Subject:
    
"""
import tensorflow as tf

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from ..utils import sequence_masking


class MultiHeadAttention(keras.layers.Layer):
    """多头注意力机制
    """

    def __init__(self,
                 n_unit,
                 n_head,
                 n_unit_each_head=None,
                 use_bias=True,
                 attention_scale=True,
                 return_attention_scores=False,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.n_unit = n_unit
        self.n_head = n_head
        self.n_unit_each_head = n_unit_each_head or n_unit // n_head
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        # weights
        self.q_dense = self.k_dense = self.v_dense = self.o_dense = None

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'n_unit': self.n_unit,
            'n_head': self.n_head,
            'n_unit_each_head': self.n_unit_each_head,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
        })
        return config

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = keras.layers.Dense(
            units=self.n_unit_each_head * self.n_head,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = keras.layers.Dense(
            units=self.n_unit_each_head * self.n_head,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = keras.layers.Dense(
            units=self.n_unit_each_head * self.n_head,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = keras.layers.Dense(
            units=self.n_unit,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        """
        query, key, value = inputs
        q_mask, v_mask = None, None
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            q_mask, v_mask = mask[0], mask[2]

        seq_len_from = tf.shape(query)[1]
        seq_len_to = tf.shape(key)[1]
        # assert K.int_shape(key)[1] == K.int_shape(value)[1]

        q = self.q_dense(query)
        k = self.k_dense(key)
        v = self.v_dense(value)

        # 维度说明：
        # B: batch_size
        # F: seq_len_from
        # T: seq_len_to
        # H: n_head
        # N: n_unit

        q = K.reshape(q, (-1, seq_len_from, self.n_head, self.n_unit_each_head))  # [B, F, H, N]
        k = K.reshape(k, (-1, seq_len_to, self.n_head, self.n_unit_each_head))  # [B, T, H, N]
        v = K.reshape(v, (-1, seq_len_to, self.n_head, self.n_unit_each_head))  # [B, T, H, N]

        q = tf.transpose(q, perm=[0, 2, 1, 3])  # [B, H, F, N]
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # [B, H, T, N]
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # [B, H, T, N]

        # 计算 attention score
        a = tf.matmul(q, k, transpose_b=True) / (self.n_unit_each_head ** 0.5)  # [B, H, F, T]
        a = sequence_masking(a, v_mask, mode='add', axis=-1)  # same
        a = K.softmax(a)  # same

        # 输出并 mask
        o = tf.matmul(a, v)  # [B, H, F, N]
        o = tf.transpose(o, perm=[0, 2, 1, 3])  # [B, F, H, N]
        o = tf.reshape(o, (-1, seq_len_from, self.n_head * self.n_unit_each_head))  # [B, F, H*N]

        o = self.o_dense(o)
        o = sequence_masking(o, q_mask, mode='mul', axis=1)  # same

        if self.return_attention_scores:
            return o, a
        return o

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.n_unit)
        if self.return_attention_scores:
            a_shape = (input_shape[0][0], self.n_head, input_shape[0][1], input_shape[1][1])
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return mask[0], None
            else:
                return mask[0]
