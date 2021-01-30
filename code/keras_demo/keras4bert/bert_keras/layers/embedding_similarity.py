#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-30 18:59
    
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


class EmbeddingSimilarity(keras.layers.Layer):
    """Calculate similarity between features and token embeddings with bias term.
    （仅用于 BERT）
    """

    def __init__(self,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param output_dim: Same as embedding output dimension.
        :param initializer: Initializer for bias.
        :param regularizer: Regularizer for bias.
        :param constraint: Constraint for bias.
        :param kwargs: Arguments for parent class.
        """
        super(EmbeddingSimilarity, self).__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.bias = None

    def get_config(self):
        config = {
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
        }
        base_config = super(EmbeddingSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(EmbeddingSimilarity, self).build(input_shape)
        self.bias = self.add_weight(
            shape=(int(input_shape[1][0]),),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            name='bias',
        )

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def call(self, inputs, **kwargs):
        inputs, embeddings = inputs
        outputs = K.bias_add(K.dot(inputs, K.transpose(embeddings)), self.bias)
        return keras.activations.softmax(outputs)
