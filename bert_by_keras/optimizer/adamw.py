#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-04-27 11:29 上午
    
Author:
    huayang
    
Subject:
    使用 tensorflow_addons.optimizers 中提供的 AdamW 无法正常收敛，原因未知；
    故参考 tf.keras 文档中自定义 Optimizer 的方法和原 bert 中 adamw 实现重写了这个版本；
        > https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
        > https://github.com/google-research/bert/blob/master/optimization.py

Notes:
    tf.keras 和 keras 中自定义 Optimizer 有区别（甚至不同版本的 tf.keras 也有区别，以下实现基于 tf 2.4）
"""
import re

import tensorflow as tf
try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

keras.optimizers.Adam._resource_scatter_add

class AdamW(keras.optimizers.Optimizer):
    """"""
    def __init__(self,
                 learning_rate=1e-5,
                 weight_decay_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 exclude_from_weight_decay=None,
                 epsilon=1e-7,
                 bias_correction=False,
                 name='AdamW',
                 **kwargs
                 ):
        """

        Args:
            learning_rate: 学习率
            weight_decay_rate: 权重衰减率
            beta_1: Adam 参数 beta_1
            beta_2: Adam 参数 beta_2
            exclude_from_weight_decay: 不应用权重衰减的参数名称列表，通过 re.search(exclude_name, var_name) 匹配
            epsilon:
            bias_correction: 是否应用 bias 修正，默认 False
            name:
            **kwargs:
        """
        super(AdamW, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('weight_decay_rate', weight_decay_rate),
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon
        self.bias_correction = bias_correction
        self.exclude_from_weight_decay = exclude_from_weight_decay or []

    def _create_slots(self, var_list):
        """"""
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None, apply_state=None):
        """"""
        var_name, var_dtype = var.name, var.dtype.base_dtype
        # 变量准备
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        t = K.cast(self.iterations + 1, var_dtype)
        lr = self._decayed_lr(var_dtype)
        wd = self._get_hyper('weight_decay_rate', var_dtype)
        beta_1 = self._get_hyper('beta_1', var_dtype)
        beta_2 = self._get_hyper('beta_2', var_dtype)

        # 梯度计算
        if indices is None:
            m_t = K.update(m, beta_1 * m + (1.0 - beta_1) * grad)
            v_t = K.update(v, beta_2 * v + (1.0 - beta_2) * K.square(grad))
        else:  # 对 sparse tensor，特别处理需要 element-wise 进行的操作
            tmp_ops = [K.update(m, beta_1 * m), K.update(v, beta_2 * v)]
            with tf.control_dependencies(tmp_ops):
                m_t = self._resource_scatter_add(m, indices, (1 - beta_1) * grad)
                v_t = self._resource_scatter_add(v, indices, (1 - beta_2) * K.square(grad))

        # bias 修正
        if self.bias_correction:  # 原 bert 中省略了以下两步
            m_t = K.update(m, m_t / (1 - K.pow(beta_1, t)))
            v_t = K.update(v, v_t / (1 - K.pow(beta_2, t)))

        delta = m_t / (K.sqrt(v_t) + self.epsilon)

        # 权重衰减
        if self._do_weight_decay(var_name):
            delta += wd * var

        var_t = var - lr * delta
        return K.update(var, var_t)

    def _resource_apply_dense(self, grad, var, apply_state):
        """"""
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        """"""
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        """"""
        config = super(AdamW, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'weight_decay_rate': self._serialize_hyperparameter('weight_decay_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'exclude_from_weight_decay': self.exclude_from_weight_decay,
            'bias_correction': self.bias_correction,
        })
        return config

    def _do_weight_decay(self, var_name):
        """"""
        return all([not re.search(exclude_name, var_name) for exclude_name in self.exclude_from_weight_decay])