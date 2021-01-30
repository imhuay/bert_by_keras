#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-30 19:52
    
Author:
    huayang
    
Subject:
    自定义 Embedding，同时返回 embedding weights
"""
try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K


class CustomEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def call(self, inputs):
        return super(CustomEmbedding, self).call(inputs), self.embeddings
