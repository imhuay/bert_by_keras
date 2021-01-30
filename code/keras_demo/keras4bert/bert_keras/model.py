#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-23 19:42
    
Author:
    huayang
    
Subject:
    
"""
import json

import numpy as np
import tensorflow as tf

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from .layers import PositionEmbedding, LayerNormalization, MultiHeadAttention, FeedForward, SegmentEmbedding
from .utils import gelu


def build_bret_from_config(config_path=None, ckpt_path=None, **kwargs):
    """"""

    def arg_replace(arg_name, arg_name_new):
        if arg_name in config:
            config[arg_name_new] = config[arg_name]
            config.pop(arg_name)

    def remove_arg(arg_name):
        config.pop(arg_name)

    config = {}
    if config_path is not None:
        config.update(json.load(open(config_path)))

    remove_arg('directionality')
    remove_arg('initializer_range')
    arg_replace('hidden_dropout_prob', 'dropout_rate')
    arg_replace('type_vocab_size', 'segment_vocab_size')
    arg_replace('num_attention_heads', 'n_attention_head')
    arg_replace('num_hidden_layers', 'n_transformer_block')
    arg_replace('intermediate_size', 'n_intermediate_unit')
    arg_replace('hidden_size', 'n_hidden_unit')
    arg_replace('attention_probs_dropout_prob', 'attention_dropout_rate')
    arg_replace('max_position_embeddings', 'max_position_len')
    config.update(kwargs)

    model = build_bert(**config)
    load_model_weights_from_checkpoint(model, config, ckpt_path)
    return model


def build_bert(n_hidden_unit=768,
               n_transformer_block=12,
               n_attention_head=12,
               n_intermediate_unit=3072,
               vocab_size=21128,
               segment_vocab_size=2,
               max_position_len=512,
               sequence_len=None,
               hidden_act=gelu,
               n_each_head_unit=None,
               embedding_size=None,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               initializer=keras.initializers.TruncatedNormal(stddev=0.02), 
               **kwargs):
    """"""
    # args assert
    embedding_size = embedding_size or n_hidden_unit
    n_each_head_unit = n_each_head_unit or n_hidden_unit // n_attention_head

    # inputs
    inputs = get_inputs(sequence_len)

    # flow
    x = apply_embeddings(inputs, vocab_size, segment_vocab_size, max_position_len, embedding_size, dropout_rate)

    for i in range(n_transformer_block):
        x = apply_main_layers(x, i,
                              n_attention_head,
                              n_each_head_unit,
                              n_hidden_unit,
                              attention_dropout_rate,
                              n_intermediate_unit,
                              hidden_act,
                              initializer)

    outputs = apply_final_layers(x)

    model = keras.Model(inputs, outputs, name='Bert')
    return model


def get_inputs(sequence_len):
    """"""
    x_in = keras.layers.Input(shape=(sequence_len,), name='Input-Token')
    s_in = keras.layers.Input(shape=(sequence_len,), name='Input-Segment')

    inputs = [x_in, s_in]
    return inputs


def apply_embeddings(inputs, vocab_size, segment_vocab_size, max_sequence_len, embedding_size, dropout_rate):
    """"""
    inputs = inputs[:]
    x, s = inputs

    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True,
                               name='Embedding-Token')(x)
    s = keras.layers.Embedding(input_dim=segment_vocab_size, output_dim=embedding_size, name='Embedding-Segment')(s)

    x = keras.layers.Add(name='Embedding-Token-Segment')([x, s])
    x = PositionEmbedding(input_dim=max_sequence_len, output_dim=embedding_size, name='Embedding-Position')(x)
    # x = keras.layers.LayerNormalization(name='Embedding-Norm')(x)
    x = LayerNormalization(name='Embedding-Norm')(x)
    x = keras.layers.Dropout(dropout_rate, name='Embedding-Dropout')(x)

    return x


def apply_main_layers(inputs, index,
                      n_attention_head,
                      n_each_head_unit,
                      n_hidden_unit,
                      dropout_rate,
                      n_intermediate_unit,
                      hidden_act,
                      initializer):
    """Att --> Add --> LN --> FFN --> Add --> LN"""
    attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
    feed_forward_name = 'Transformer-%d-FeedForward' % index

    x = inputs
    xi = x
    x = MultiHeadAttention(heads=n_attention_head,
                           head_size=n_each_head_unit,
                           out_dim=n_hidden_unit,
                           name=attention_name)([x, x, x])
    x = keras.layers.Dropout(dropout_rate, name='%s-Dropout' % attention_name)(x)
    x = keras.layers.Add(name='%s-Add' % attention_name)([xi, x])
    # x = keras.layers.LayerNormalization(name='%s-Norm' % attention_name)(x)
    x = LayerNormalization(name='%s-Norm' % attention_name)(x)

    xi = x
    x = FeedForward(units=n_intermediate_unit, activation=hidden_act,
                    kernel_initializer=initializer, name=feed_forward_name)(x)
    x = keras.layers.Dropout(dropout_rate, name='%s-Dropout' % feed_forward_name)(x)
    x = keras.layers.Add(name='%s-Add' % feed_forward_name)([xi, x])
    # x = keras.layers.LayerNormalization(name='%s-Norm' % feed_forward_name)(x)
    x = LayerNormalization(name='%s-Norm' % feed_forward_name)(x)

    return x


def apply_final_layers(inputs):
    """"""
    x = inputs
    outputs = [x]

    if len(outputs) == 1:
        outputs = outputs[0]
    elif len(outputs) == 2:
        outputs = outputs[1]
    else:
        outputs = outputs[1:]

    return outputs


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       training=False):
    """Load trained official model from checkpoint.

    :param model: Built keras model.
    :param config: Loaded configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    """
    loader = checkpoint_loader(checkpoint_file)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('bert/embeddings/position_embeddings')[:config['max_position_len'], :],
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ])
    for i in range(config['n_transformer_block']):
        # try:
        #     model.get_layer(name='Transformer-%d-MultiHeadSelfAttention' % i)
        # except ValueError as e:
        #     continue
        model.get_layer(name='Transformer-%d-MultiHeadSelfAttention' % i).set_weights([
            loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
            loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
            loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
            loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
        ])
        model.get_layer(name='Transformer-%d-MultiHeadSelfAttention-Norm' % i).set_weights([
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='Transformer-%d-FeedForward' % i).set_weights([
            loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
            loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
            loader('bert/encoder/layer_%d/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/output/dense/bias' % i),
        ])
        model.get_layer(name='Transformer-%d-FeedForward-Norm' % i).set_weights([
            loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
        ])

    if training:
        model.get_layer(name='MLM-Dense').set_weights([
            loader('cls/predictions/transform/dense/kernel'),
            loader('cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='MLM-Norm').set_weights([
            loader('cls/predictions/transform/LayerNorm/gamma'),
            loader('cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='MLM-Sim').set_weights([
            loader('cls/predictions/output_bias'),
        ])
        model.get_layer(name='NSP-Dense').set_weights([
            loader('bert/pooler/dense/kernel'),
            loader('bert/pooler/dense/bias'),
        ])
        model.get_layer(name='NSP').set_weights([
            np.transpose(loader('cls/seq_relationship/output_weights')),
            loader('cls/seq_relationship/output_bias'),
        ])


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader
