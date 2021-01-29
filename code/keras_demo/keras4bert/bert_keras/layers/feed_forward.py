#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-24 20:27
    
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

from ..utils import gelu

# 低版本 keras 不含 gelu 激活函数
try:
    keras.activations.get('gelu')
except:
    keras.utils.get_custom_objects()['gelu'] = gelu


class FeedForward(keras.layers.Layer):
    """FeedForward层
    如果activation不是一个list，那么它就是两个Dense层的叠加；如果activation是
    一个list，那么第一个Dense层将会被替换成门控线性单元（Gated Linear Unit）。
    参考论文: https://arxiv.org/abs/2002.05202
    """

    def __init__(
            self,
            units,
            activation='gelu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        if not isinstance(activation, list):
            activation = [activation]
        self.activation = [keras.activations.get(act) for act in activation]
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        for i, activation in enumerate(self.activation):
            i_dense = keras.layers.Dense(
                units=self.units,
                activation=activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, 'i%s_dense' % i, i_dense)
        # weights
        self.o_dense = None

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)

        output_dim = input_shape[-1]
        self.o_dense = keras.layers.Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, **kwargs):
        x = self.i0_dense(inputs)
        for i in range(1, len(self.activation)):
            x = x * getattr(self, 'i%s_dense' % i)(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [
                keras.activations.serialize(act) for act in self.activation
            ],
            'use_bias': self.use_bias,
            'kernel_initializer':
                keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
