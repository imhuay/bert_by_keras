#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-27 17:55
    
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


class SegmentEmbedding(keras.layers.Embedding):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.supports_masking = True
