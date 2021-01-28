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


class MultiHeadAttention(keras.layers.Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        heads,
        head_size,
        out_dim=None,
        use_bias=True,
        attention_scale=True,
        return_attention_scores=False,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        # weights
        self.q_dense = self.k_dense = self.v_dense = self.o_dense = None

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = keras.layers.Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = keras.layers.Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = keras.layers.Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = keras.layers.Dense(
            units=self.out_dim,
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

        q = K.reshape(q, (-1, seq_len_from, self.heads, self.head_size))  # [batch_size, seq_len_from, n_head, n_unit]
        k = K.reshape(k, (-1, seq_len_to, self.heads, self.head_size))  # [batch_size, seq_len_to, n_head, n_unit]
        v = K.reshape(v, (-1, seq_len_to, self.heads, self.head_size))  # [batch_size, seq_len_to, n_head, n_unit]

        q = tf.transpose(q, perm=[0, 2, 1, 3])  # [batch_size, n_head, seq_len_from, n_unit]
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # [batch_size, n_head, seq_len_to, n_unit]
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # [batch_size, n_head, seq_len_to, n_unit]

        # 计算 attention score
        a = tf.matmul(q, k, transpose_b=True) / (self.head_size ** 0.5)  # [batch_size, n_head, seq_len_from, seq_len_to]
        a = sequence_masking(a, v_mask, mode='add', axis=-1)  # same
        a = K.softmax(a)  # same

        # 输出并 mask
        o = tf.matmul(a, v)  # [batch_size, n_head, seq_len_from, n_unit]
        o = tf.transpose(o, perm=[0, 2, 1, 3])  # [batch_size, seq_len_from, n_head, n_unit]
        o = tf.reshape(o, (-1, seq_len_from, self.heads * self.head_size))  # [batch_size, seq_len_from, n_head*n_unit]

        o = self.o_dense(o)
        o = sequence_masking(o, q_mask, mode='mul', axis=1)  # same

        if self.return_attention_scores:
            return o, a
        return o

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.heads, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sequence_masking(x, mask, mode='add', axis=-1):
    """序列 mask

    Args:
        x: 2D 或 2D 以上张量，如 [batch_size, seq_len, input_size]
        mask: 形如  (batch_size, seq_len) 的 0/1矩阵
        mode: 有 'mul' 和 'add' 两种：
            mul 会将 pad 部分置零，一般用于全连接层之前；
            add 会把 pad 部分减去一个大的常数，一般用于 softmax 之前。
        axis: 需要 mask 的序列所在轴

    Returns:
        tensor with shape same as x
    """
    if mask is None:
        return x

    assert mode in ['add', 'mul'], mode

    if axis < 0:
        axis = K.ndim(x) + axis

    for _ in range(axis - 1):
        mask = K.expand_dims(mask, axis=1)

    for _ in range(K.ndim(x) - axis - 1):
        mask = K.expand_dims(mask, axis=-1)

    if mode == 'mul':
        return x * mask
    return x - (1 - mask) * 1e12
