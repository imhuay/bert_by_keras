#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-23 20:25
    
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


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_mask(self, inputs, mask=None):
        return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs):
        return [super(TokenEmbedding, self).call(inputs), self.embeddings + 0]
