#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-23 20:22
    
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


class PositionEmbedding(keras.layers.Layer):
    """
    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 input_dim,
                 output_dim,
                 mode=MODE_ADD,
                 embedding_initializer='uniform',
                 embedding_regularizer=None,
                 embedding_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
        :param input_dim: The maximum absolute value of positions.
        :param output_dim: The embedding dimension.
        :param embedding_initializer:
        :param embedding_regularizer:
        :param activity_regularizer:
        :param embedding_constraint:
        :param kwargs:
        """
        super(PositionEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embedding_initializer = keras.initializers.get(embedding_initializer)
        self.embedding_regularizer = keras.regularizers.get(embedding_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.embedding_constraint = keras.constraints.get(embedding_constraint)
        # weight
        self.embeddings = None

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'mode': self.mode,
            'embedding_initializer': keras.initializers.serialize(self.embedding_initializer),
            'embedding_regularizer': keras.regularizers.serialize(self.embedding_regularizer),
            'embedding_constraint': keras.constraints.serialize(self.embedding_constraint),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embedding_initializer,
            name='embeddings',
            regularizer=self.embedding_regularizer,
            constraint=self.embedding_constraint,
        )
        super(PositionEmbedding, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]

        pos_embeddings = K.tile(
            K.expand_dims(self.embeddings[:seq_len, :self.output_dim], axis=0),
            [batch_size, 1, 1],
        )

        if self.mode == self.MODE_CONCAT:
            return K.concatenate([inputs, pos_embeddings], axis=-1)
        return inputs + pos_embeddings
